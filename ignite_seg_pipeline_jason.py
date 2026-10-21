"""
IGNITE H&E | BASTNet (Boundary-aware, Attention-gated, Stain-invariant, Transformer hybrid)
=========================================================================================
16-class tissue segmentation (he_label_map.json; 0 = Unannotated is ignored) + 3-way subtype classification
(adenocarcinoma / adenosquamous / squamous_cell_carcinoma; label = folder name).

Single script:
  1. Patient-level, subtype-stratified 70/20/10 split (no patient leakage)
  2. "AI Pathologist" = out-of-fold K-fold ensemble trained on pathologist masks -> AI ground truth
  3. Model-P (trained on pathologist masks) and Model-AI (trained on AI ground truth)
  4. Patient-grouped K-fold CV, Wilcoxon signed-rank, IQR/box/violin plots
  5. t-SNE, confusion matrices, ROC
  6. Figure: Original | Pathologist GT | AI GT | Pred (pathologist-trained) | Pred (AI-trained) | Uncertainty

RTX 5080 (Blackwell) needs a CUDA 12.8+ PyTorch build:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install timm opencv-python numpy pandas scikit-learn scipy matplotlib seaborn tqdm
"""
import os, re, glob, random, json, warnings
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, timm
from tqdm import tqdm
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, f1_score, accuracy_score)
from sklearn.preprocessing import label_binarize
warnings.filterwarnings("ignore")

# ============================== CONFIG ==============================
ROOT     = r"E:\ignite tool kit he segmentation\Data\organised_he"
SUBTYPES = ["adenocarcinoma", "adenosquamous", "squamous_cell_carcinoma"]   # = classification classes
NUM_CLS  = len(SUBTYPES)
LABELS = {"Unannotated": 0, "Background": 1, "Tumor epithelium": 2, "Reactive epithelium": 3, "Stroma": 4,
          "Inflammation": 5, "Alveolar tissue": 6, "Fatty tissue": 7, "Necrotic tissue": 8, "Erythrocytes": 9,
          "Bronchial epithelium": 10, "Mucus/Plasma/Fluids": 11, "Cartilage/Bone": 12, "Macrophages": 13,
          "Muscle": 14, "Liver": 15, "Keratinization": 16}
for _p in (os.path.join(ROOT, "he_label_map.json"), "he_label_map.json"):     # use the official file if present
    if os.path.exists(_p): LABELS = json.load(open(_p)); print("Loaded label map:", _p); break
NUM_SEG   = max(LABELS.values()) + 1                     # 17 channels; channel 0 (Unannotated) is ignored
SEG_NAMES = [n for n, _ in sorted(LABELS.items(), key=lambda kv: kv[1])]
TUMOR_ID  = LABELS["Tumor epithelium"]
MASK_TOKENS = ("mask", "annot", "label", "gt", "seg")   # how masks are recognised (folder or file name)
PATIENT_REGEX = r"^([^_\.\-]+)"      # patient ID = first token of filename  (CHECK: should give 155 patients)
EXPECTED_PATIENTS = 155
IMG_SIZE = 384                       # multiple of 32
BATCH    = 24
EPOCHS, EPOCHS_CV, K_FOLDS = 60, 25, 5
LR, SEED, PRETRAINED = 2e-4, 42, True
WORKERS  = 4
OUT      = "results"
DEV      = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ============================== DATA ==============================
EXTS = ("png", "jpg", "jpeg", "tif", "tiff", "bmp")
def imread(p, flag=cv2.IMREAD_UNCHANGED): return cv2.imdecode(np.fromfile(p, np.uint8), flag)
def strip_key(stem): return re.sub(r"[_\-\.]?(masks?|annotations?|annot|labels?|gt|seg)$", "", stem, flags=re.I).lower()

def load_all():
    X, Y, C, P, N = [], [], [], [], []
    for ci, sub in enumerate(SUBTYPES):
        base = os.path.join(ROOT, sub)
        files = [p for e in EXTS for p in glob.glob(os.path.join(base, "**", f"*.{e}"), recursive=True)]
        files += [p for e in EXTS for p in glob.glob(os.path.join(base, "**", f"*.{e.upper()}"), recursive=True)]
        files = sorted(set(files))
        is_mask = lambda p: any(t in os.path.relpath(p, base).lower() for t in MASK_TOKENS)
        masks = {(strip_key(os.path.splitext(os.path.basename(p))[0])): p for p in files if is_mask(p)}
        imgs = [p for p in files if not is_mask(p)]
        used = 0
        for p in tqdm(imgs, desc=sub):
            stem = os.path.splitext(os.path.basename(p))[0]
            m = masks.get(strip_key(stem))
            if m is None: continue
            im = cv2.cvtColor(imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            ms = imread(m)
            if ms.ndim == 3:
                assert (ms[..., 0] == ms[..., 1]).all() and (ms[..., 0] == ms[..., 2]).all(), \
                    f"{m} looks like an RGB colour mask; masks must store label IDs (0-16)"
                ms = ms[..., 0]
            im = cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            ms = cv2.resize(ms, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            X.append(im); Y.append(np.clip(ms, 0, NUM_SEG - 1).astype(np.uint8)); C.append(ci); N.append(stem)
            P.append(re.match(PATIENT_REGEX, stem).group(1)); used += 1
        print(f"  {sub}: {len(imgs)} images, {len(masks)} masks, {used} matched pairs")
    assert len(X) > 0, "No image/mask pairs matched. Check folder layout / MASK_TOKENS."
    X, Y, C, P, N = np.stack(X), np.stack(Y), np.array(C), np.array(P), np.array(N)
    frac = np.bincount(Y.ravel(), minlength=NUM_SEG) / Y.size
    print("Pixel fraction per class:", {SEG_NAMES[i]: round(float(f), 4) for i, f in enumerate(frac)})
    print(f"Total {len(X)} pairs | {len(set(P))} unique patient IDs (expected {EXPECTED_PATIENTS})")
    if len(set(P)) != EXPECTED_PATIENTS: print("  !! Patient count differs - adjust PATIENT_REGEX so the split stays leak-free.")
    return X, Y, C, P, N

def stain_jitter(img, s=0.25):
    """Random perturbation in optical-density space (stain/scanner variability)."""
    od = -np.log((img.astype(np.float32) + 1) / 256.)
    od = od * np.random.uniform(1 - s, 1 + s, (1, 1, 3)) + np.random.uniform(-s / 5, s / 5, (1, 1, 3))
    return np.clip(256 * np.exp(-od) - 1, 0, 255).astype(np.uint8)

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
def to_tensor(img): return torch.from_numpy(((img / 255. - MEAN) / STD).transpose(2, 0, 1).astype(np.float32))

class DS(torch.utils.data.Dataset):
    def __init__(self, X, Y, C): self.X, self.Y, self.C = X, Y, C
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        im, ms = self.X[i], self.Y[i]
        k = np.random.randint(4); im, ms = np.rot90(im, k), np.rot90(ms, k)
        if np.random.rand() < .5: im, ms = im[:, ::-1], ms[:, ::-1]
        im, ms = np.ascontiguousarray(im), np.ascontiguousarray(ms)
        a = stain_jitter(im, .15); b = stain_jitter(im, .35)          # 2 stain views -> consistency loss
        return to_tensor(a), to_tensor(b), torch.from_numpy(ms.astype(np.int64)), int(self.C[i])

# ============================== NOVEL MODEL ==============================
class ConvBR(nn.Sequential):
    def __init__(s, i, o, k=3, d=1): super().__init__(nn.Conv2d(i, o, k, padding=d * (k // 2), dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

class ASPP(nn.Module):                                       # multi-scale tissue context
    def __init__(s, i, o):
        super().__init__()
        s.b = nn.ModuleList([ConvBR(i, o, 1), ConvBR(i, o, 3, 2), ConvBR(i, o, 3, 4)])
        s.g = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(i, o, 1), nn.ReLU(True)); s.p = ConvBR(o * 4, o, 1)
    def forward(s, x):
        g = s.g(x).expand(-1, -1, x.shape[2], x.shape[3])
        return s.p(torch.cat([b(x) for b in s.b] + [g], 1))

class GlobalTransformer(nn.Module):                          # long-range context
    def __init__(s, d, L=2, heads=8):
        super().__init__()
        s.pos = nn.Conv2d(d, d, 3, padding=1, groups=d)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, d * 2, 0.1, batch_first=True, norm_first=True), L)
    def forward(s, x):
        x = x + s.pos(x); B, C, H, W = x.shape
        return s.enc(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)

class AttGate(nn.Module):
    def __init__(s, g, x, m):
        super().__init__()
        s.wg = nn.Conv2d(g, m, 1); s.wx = nn.Conv2d(x, m, 1); s.psi = nn.Sequential(nn.Conv2d(m, 1, 1), nn.Sigmoid())
    def forward(s, g, x): return x * s.psi(F.relu(s.wg(g) + s.wx(x)))

class Up(nn.Module):
    def __init__(s, gc, sc, oc):
        super().__init__()
        s.att = AttGate(gc, sc, max(sc // 2, 16)); s.c = nn.Sequential(ConvBR(gc + sc, oc), ConvBR(oc, oc), nn.Dropout2d(0.1))
    def forward(s, g, skip):
        g = F.interpolate(g, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return s.c(torch.cat([g, s.att(g, skip)], 1))

class BASTNet(nn.Module):
    """ConvNeXt-T encoder | ASPP + Transformer bottleneck | attention-gated decoder |
       heads: tumour segmentation, boundary, subtype classification."""
    def __init__(s, nseg=NUM_SEG, ncls=NUM_CLS, pretrained=PRETRAINED):
        super().__init__()
        s.enc = timm.create_model("convnext_tiny", pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        ch = s.enc.feature_info.channels()
        s.aspp = ASPP(ch[3], 256); s.trf = GlobalTransformer(256)
        s.u3 = Up(256, ch[2], 192); s.u2 = Up(192, ch[1], 128); s.u1 = Up(128, ch[0], 64)
        s.seg = nn.Sequential(ConvBR(64, 32), nn.Conv2d(32, nseg, 1))
        s.bnd = nn.Sequential(ConvBR(64, 32), nn.Conv2d(32, 1, 1))
        s.cls = nn.Sequential(nn.Linear(256 + 64, 128), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(128, ncls))
    def forward(s, x):
        H, W = x.shape[2:]
        f = s.enc(x); b = s.trf(s.aspp(f[3]))
        d = s.u1(s.u2(s.u3(b, f[2]), f[1]), f[0])
        up = lambda t: F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        return up(s.seg(d)), up(s.bnd(d)), s.cls(torch.cat([b.mean((2, 3)), d.mean((2, 3))], 1)), b.mean((2, 3))

def boundary_target(m):
    m = m.float().unsqueeze(1)                                       # boundary = label changes within 5x5 window
    return ((F.max_pool2d(m, 5, 1, 2) != -F.max_pool2d(-m, 5, 1, 2))).float()

def dice_loss(lg, y):
    valid = (y > 0).unsqueeze(1).float()                             # ignore Unannotated
    p = lg.softmax(1) * valid; oh = F.one_hot(y, NUM_SEG).permute(0, 3, 1, 2).float() * valid
    i = (p * oh).sum((0, 2, 3)); u = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    present = oh.sum((0, 2, 3)) > 0; present[0] = False
    return 1 - ((2 * i + 1) / (u + 1))[present].mean() if present.any() else lg.sum() * 0

def focal_ce(lg, y, g=2.0):
    ce = F.cross_entropy(lg, y, reduction="none", ignore_index=0)
    v = (y > 0).float(); return ((((1 - torch.exp(-ce)) ** g) * ce) * v).sum() / v.sum().clamp(min=1)

# ============================== TRAIN / PREDICT ==============================
@torch.no_grad()
def predict(models, X, tta=True, bs=32, feats=False):
    models = models if isinstance(models, list) else [models]
    for m in models: m.eval()
    PS, CS, FS = [], [], []
    for i in range(0, len(X), bs):
        x = torch.stack([to_tensor(a) for a in X[i:i + bs]]).to(DEV)
        ps = cs = fs = 0; n = 0
        for m in models:
            for fl in ([[], [3], [2]] if tta else [[]]):
                s, _, c, f = m(x.flip(fl) if fl else x)
                s = s.softmax(1); s = s.flip(fl) if fl else s
                ps, cs, fs, n = ps + s, cs + c.softmax(1), fs + f, n + 1
        PS.append((ps / n).cpu()); CS.append((cs / n).cpu()); FS.append((fs / n).cpu())
    ps_all = torch.cat(PS).numpy(); ps_all[:, 0] = 0                 # 'Unannotated' is never a prediction
    out = (ps_all, torch.cat(CS).numpy())
    return out + (torch.cat(FS).numpy(),) if feats else out

@torch.no_grad()
def mc_uncertainty(model, X, T=8, bs=16):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)): m.train()
    U = []
    for i in range(0, len(X), bs):
        x = torch.stack([to_tensor(a) for a in X[i:i + bs]]).to(DEV)
        p = torch.stack([model(x)[0].softmax(1) for _ in range(T)]).mean(0)
        U.append((-(p * (p + 1e-8).log()).sum(1)).cpu())
    model.eval(); return torch.cat(U).numpy()

def per_image_metrics(pred, gt):
    """Multi-class Dice/IoU on annotated pixels only (GT>0). Columns: Dice, IoU (macro over classes present),
       PixAcc, Dice_Tumor, and D_<class> per-class Dice (NaN if class absent in both)."""
    rows = []
    for p, g in zip(pred, gt):
        valid = g > 0; d, j, r = [], [], {}
        for c in range(1, NUM_SEG):
            a, b = (p == c) & valid, g == c
            if a.sum() + b.sum() == 0: r[f"D_{SEG_NAMES[c]}"] = np.nan; continue
            inter = (a & b).sum(); dc = 2 * inter / (a.sum() + b.sum())
            d.append(dc); j.append(inter / (a | b).sum()); r[f"D_{SEG_NAMES[c]}"] = dc
        rows.append(dict(Dice=np.mean(d) if d else np.nan, IoU=np.mean(j) if j else np.nan,
                         PixAcc=(p[valid] == g[valid]).mean() if valid.any() else np.nan,
                         Dice_Tumor=r[f"D_{SEG_NAMES[TUMOR_ID]}"], **r))
    return pd.DataFrame(rows)

def train(Xtr, Ytr, Ctr, Xva, Yva, Cva, epochs, tag):
    dl = torch.utils.data.DataLoader(DS(Xtr, Ytr, Ctr), BATCH, shuffle=True, num_workers=WORKERS,
                                     persistent_workers=WORKERS > 0, drop_last=len(Xtr) > BATCH, pin_memory=True)
    model = BASTNet().to(DEV).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=epochs * len(dl))
    cw = torch.tensor(len(Ctr) / (NUM_CLS * np.maximum(np.bincount(Ctr, minlength=NUM_CLS), 1)), dtype=torch.float32, device=DEV)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda"); best, path = -1, f"{OUT}/{tag}.pt"
    for ep in range(epochs):
        model.train(); tl = 0
        for x, x2, y, c in dl:
            x, x2, y, c = x.to(DEV, non_blocking=True), x2.to(DEV, non_blocking=True), y.to(DEV), c.to(DEV)
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                s, b, cl, f1 = model(x); _, _, _, f2 = model(x2)
                loss = (focal_ce(s, y) + dice_loss(s, y)
                        + 0.5 * F.binary_cross_entropy_with_logits(b, boundary_target(y))
                        + 0.5 * F.cross_entropy(cl, c, weight=cw)
                        + 0.1 * F.mse_loss(F.normalize(f1.float(), dim=1), F.normalize(f2.float(), dim=1)))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); tl += loss.item()
        if len(Xva):
            pr, cp = predict(model, Xva, tta=False); d = per_image_metrics(pr.argmax(1), Yva).Dice.mean(); acc = accuracy_score(Cva, cp.argmax(1))
            score = d + 0.5 * acc
            if score > best: best = score; torch.save(model.state_dict(), path)
            print(f"[{tag}] ep{ep + 1}/{epochs} loss {tl / len(dl):.4f} | val Dice {d:.4f} cls-acc {acc:.4f}")
        else: torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load(path)); return model.eval()

# ============================== MAIN ==============================
if __name__ == "__main__":
    seed_all(SEED); print("Device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
    torch.backends.cudnn.benchmark = True
    X, Y, C, P, N = load_all()

    # ---- 70/20/10 patient-level, subtype-stratified split ----
    folds = list(StratifiedGroupKFold(10, shuffle=True, random_state=SEED).split(X, C, P))
    te = folds[0][1]; rest = folds[0][0]
    f9 = list(StratifiedGroupKFold(9, shuffle=True, random_state=SEED).split(X[rest], C[rest], P[rest]))
    va = rest[np.concatenate([f9[0][1], f9[1][1]])]; tr = np.setdiff1d(rest, va)          # 2/9 of 90% = 20%
    assert not (set(P[tr]) & set(P[va]) or set(P[tr]) & set(P[te]) or set(P[va]) & set(P[te])), "patient leakage"
    sp = np.array(["train"] * len(X), dtype=object); sp[va] = "val"; sp[te] = "test"
    pd.DataFrame({"name": N, "patient": P, "subtype": [SUBTYPES[c] for c in C], "split": sp}).to_csv(f"{OUT}/split.csv", index=False)
    print(pd.crosstab(sp, [SUBTYPES[c] for c in C]))

    # ---- Patient-grouped K-fold CV + AI PATHOLOGIST (out-of-fold ensemble) ----
    AI_Y = np.zeros_like(Y); fold_models, cv_rows = [], []
    for k, (a, b) in enumerate(StratifiedGroupKFold(K_FOLDS, shuffle=True, random_state=SEED).split(tr, C[tr], P[tr])):
        m = train(X[tr[a]], Y[tr[a]], C[tr[a]], X[tr[b]], Y[tr[b]], C[tr[b]], EPOCHS_CV, f"cv_fold{k}")
        pr, cp = predict(m, X[tr[b]]); AI_Y[tr[b]] = pr.argmax(1)            # OOF -> honest AI annotation
        d = per_image_metrics(pr.argmax(1), Y[tr[b]]); d["fold"] = k + 1
        d["cls_acc"] = accuracy_score(C[tr[b]], cp.argmax(1)); d["cls_f1"] = f1_score(C[tr[b]], cp.argmax(1), average="macro")
        cv_rows.append(d); fold_models.append(m)
    cv = pd.concat(cv_rows); cv.to_csv(f"{OUT}/cv_metrics.csv", index=False)
    fm = cv.groupby("fold")[["Dice", "IoU", "PixAcc", "cls_acc", "cls_f1"]].mean()
    print(fm.round(4)); print(fm.agg(["mean", "std"]).round(4)); fm.agg(["mean", "std"]).to_csv(f"{OUT}/cv_summary.csv")
    for idx in (va, te): AI_Y[idx] = predict(fold_models, X[idx])[0].argmax(1)  # AI-pathologist consensus
    np.save(f"{OUT}/ai_groundtruth.npy", AI_Y)

    # ---- Final models ----
    model_P  = train(X[tr], Y[tr],    C[tr], X[va], Y[va],    C[va], EPOCHS, "model_pathologist")
    model_AI = train(X[tr], AI_Y[tr], C[tr], X[va], AI_Y[va], C[va], EPOCHS, "model_AI")

    # ---- Test evaluation ----
    pP, cP, _ = predict(model_P, X[te], feats=True); pA, cA, _ = predict(model_AI, X[te], feats=True)
    prP, prA = pP.argmax(1), pA.argmax(1)
    mP, mA = per_image_metrics(prP, Y[te]), per_image_metrics(prA, Y[te])
    mPa, mAa = per_image_metrics(prP, AI_Y[te]), per_image_metrics(prA, AI_Y[te])
    mAIp = per_image_metrics(AI_Y[te], Y[te])
    res = pd.DataFrame({"image": N[te], "patient": P[te], "subtype": [SUBTYPES[c] for c in C[te]],
        "Dice_P_vs_Path": mP.Dice, "Dice_AI_vs_Path": mA.Dice, "IoU_P_vs_Path": mP.IoU, "IoU_AI_vs_Path": mA.IoU,
        "TumDice_P_vs_Path": mP.Dice_Tumor, "TumDice_AI_vs_Path": mA.Dice_Tumor,
        "Dice_P_vs_AI": mPa.Dice, "Dice_AI_vs_AI": mAa.Dice, "Dice_AIpath_vs_Path": mAIp.Dice})
    res.to_csv(f"{OUT}/test_per_image.csv", index=False)
    print(res.iloc[:, 3:].agg(["mean", "std", "median"]).round(4))
    pc = pd.concat([mP.filter(like="D_").add_prefix("P|"), mA.filter(like="D_").add_prefix("AI|")], axis=1)
    pc.insert(0, "subtype", res.subtype.values); pc.to_csv(f"{OUT}/test_per_class_dice.csv", index=False)
    print(pc.drop(columns="subtype").mean().round(3).to_string())

    # ---- Statistics (paired Wilcoxon signed-rank + IQR) ----
    stats = []
    tests = [("Dice: Model-P vs Model-AI (vs pathologist GT)", "Dice_P_vs_Path", "Dice_AI_vs_Path"),
             ("IoU : Model-P vs Model-AI (vs pathologist GT)", "IoU_P_vs_Path", "IoU_AI_vs_Path"),
             ("Dice: Model-P vs AI-pathologist mask (vs human GT)", "Dice_P_vs_Path", "Dice_AIpath_vs_Path"),
             ("Tumor-epithelium Dice: Model-P vs Model-AI", "TumDice_P_vs_Path", "TumDice_AI_vs_Path")]
    for st in [None] + SUBTYPES:
        r = res if st is None else res[res.subtype == st]
        for name, ca, cb in ([tests[0], tests[3]] if st else tests):
            rr = r[[ca, cb]].dropna()
            try: w, p = wilcoxon(rr[ca], rr[cb])
            except ValueError: w, p = np.nan, np.nan
            q = lambda s: np.round([s.quantile(.25), s.median(), s.quantile(.75)], 4).tolist()
            stats.append(dict(subset=st or "all", test=name, n=len(rr), W=w, p=p, A_Q1_med_Q3=q(r[ca]), B_Q1_med_Q3=q(r[cb])))
    pd.DataFrame(stats).to_csv(f"{OUT}/wilcoxon_stats.csv", index=False); print(pd.DataFrame(stats).to_string())

    # ---- Classification (3 subtypes) ----
    cls_out = {}
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    for i, (tag, cp) in enumerate((("Model-P", cP), ("Model-AI", cA))):
        yt, yp = C[te], cp.argmax(1)
        cls_out[tag] = dict(acc=accuracy_score(yt, yp), f1_macro=f1_score(yt, yp, average="macro"))
        try: cls_out[tag]["auc_macro_ovr"] = roc_auc_score(yt, cp, multi_class="ovr", labels=range(NUM_CLS))
        except Exception: pass
        print(tag, classification_report(yt, yp, target_names=SUBTYPES, labels=range(NUM_CLS), zero_division=0))
        sns.heatmap(confusion_matrix(yt, yp, labels=range(NUM_CLS)), annot=True, fmt="d", cmap="Blues", xticklabels=SUBTYPES, yticklabels=SUBTYPES, ax=ax[0, i])
        ax[0, i].set_title(f"Confusion matrix - {tag}"); ax[0, i].tick_params(axis="x", rotation=20)
        yb = label_binarize(yt, classes=range(NUM_CLS))
        for c in range(NUM_CLS):
            if yb[:, c].sum() and (1 - yb[:, c]).sum():
                fpr, tpr, _ = roc_curve(yb[:, c], cp[:, c]); ax[1, i].plot(fpr, tpr, label=f"{SUBTYPES[c]} AUC={roc_auc_score(yb[:, c], cp[:, c]):.3f}")
        ax[1, i].plot([0, 1], [0, 1], "k--"); ax[1, i].legend(); ax[1, i].set_title(f"ROC (one-vs-rest) - {tag}")
    plt.tight_layout(); plt.savefig(f"{OUT}/classification.png", dpi=200); plt.close()
    json.dump(cls_out, open(f"{OUT}/classification_metrics.json", "w"), indent=2)

    # ---- IQR box / violin / paired / per-subtype ----
    long = res.melt(id_vars=["image", "subtype"], value_vars=["Dice_P_vs_Path", "Dice_AI_vs_Path", "Dice_AIpath_vs_Path"], var_name="Model", value_name="Dice")
    long["Model"] = long.Model.map({"Dice_P_vs_Path": "Model-P", "Dice_AI_vs_Path": "Model-AI", "Dice_AIpath_vs_Path": "AI-pathologist mask"})
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.5))
    sns.boxplot(data=long, x="Model", y="Dice", ax=ax[0], showmeans=True, whis=1.5); sns.stripplot(data=long, x="Model", y="Dice", ax=ax[0], color="k", size=2, alpha=.4)
    ax[0].set_title("Test Dice: median & IQR (box), 1.5xIQR whiskers"); ax[0].tick_params(axis="x", rotation=15)
    sns.violinplot(data=long, x="Model", y="Dice", inner="quartile", ax=ax[1]); ax[1].set_title("Distribution (quartiles)"); ax[1].tick_params(axis="x", rotation=15)
    ax[2].scatter(res.Dice_P_vs_Path, res.Dice_AI_vs_Path, s=14, c=pd.factorize(res.subtype)[0], cmap="tab10"); ax[2].plot([0, 1], [0, 1], "r--")
    ax[2].set_xlabel("Dice Model-P"); ax[2].set_ylabel("Dice Model-AI"); ax[2].set_title(f"Paired (Wilcoxon p={stats[0]['p']:.3g})")
    sns.boxplot(data=long[long.Model != "AI-pathologist mask"], x="subtype", y="Dice", hue="Model", ax=ax[3]); ax[3].set_title("Dice by subtype (IQR)"); ax[3].tick_params(axis="x", rotation=15)
    plt.tight_layout(); plt.savefig(f"{OUT}/iqr_wilcoxon_plots.png", dpi=200); plt.close()
    plt.figure(figsize=(7, 4.5)); sns.boxplot(data=cv, x="fold", y="Dice"); plt.title(f"Patient-grouped {K_FOLDS}-fold CV (Dice, IQR)")
    plt.tight_layout(); plt.savefig(f"{OUT}/cv_boxplot.png", dpi=200); plt.close()

    pcl = pc.melt(id_vars="subtype", var_name="k", value_name="Dice").dropna()
    pcl["Model"] = pcl.k.str.split("|").str[0].map({"P": "Model-P", "AI": "Model-AI"}); pcl["Class"] = pcl.k.str.split("|").str[1].str[2:]
    plt.figure(figsize=(17, 6)); sns.boxplot(data=pcl, x="Class", y="Dice", hue="Model", whis=1.5)
    plt.xticks(rotation=45, ha="right"); plt.title("Per-class Dice on test set (median & IQR)"); plt.tight_layout()
    plt.savefig(f"{OUT}/per_class_dice_iqr.png", dpi=200); plt.close()

    # ---- t-SNE of bottleneck features (val + test) ----
    idx = np.concatenate([va, te]); split_lab = np.array(["val"] * len(va) + ["test"] * len(te))
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.8))
    for i, (mdl, t) in enumerate(((model_P, "Model-P"), (model_AI, "Model-AI"))):
        f = predict(mdl, X[idx], feats=True)[2]
        z = PCA(min(50, len(f) - 1)).fit_transform(f)
        z = TSNE(2, perplexity=min(30, max(5, len(f) // 4)), random_state=SEED, init="pca").fit_transform(z)
        sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=[SUBTYPES[c] for c in C[idx]], style=split_lab, ax=ax[i], s=40); ax[i].set_title(f"t-SNE features - {t}")
        if i == 0: zz = z
    sns.scatterplot(x=zz[:, 0], y=zz[:, 1], hue=P[idx], legend=False, ax=ax[2], s=30, palette="tab20"); ax[2].set_title("t-SNE Model-P coloured by patient")
    plt.tight_layout(); plt.savefig(f"{OUT}/tsne.png", dpi=200); plt.close()

    # ---- Qualitative figure: 2 test examples per subtype ----
    cols_rgb = plt.cm.tab20(np.linspace(0, 1, 20))[:NUM_SEG].copy(); cols_rgb[0] = [0, 0, 0, 1]
    cmap = matplotlib.colors.ListedColormap(cols_rgb)
    show = [i for c in range(NUM_CLS) for i in np.where(C[te] == c)[0][:2]]
    U = mc_uncertainty(model_P, X[te][show])
    cols = ["Original H&E", "Pathologist GT", "AI-pathologist GT", "Pred (trained on pathologist)", "Pred (trained on AI GT)", "Uncertainty (MC-dropout)"]
    fig, ax = plt.subplots(len(show), 6, figsize=(21, 3.5 * len(show)))
    for r, i in enumerate(show):
        ims = [X[te][i], Y[te][i], AI_Y[te][i], prP[i], prA[i], U[r]]
        for c, im in enumerate(ims):
            kw = {} if c == 0 else dict(cmap="magma") if c == 5 else dict(cmap=cmap, vmin=0, vmax=NUM_SEG - 1, interpolation="nearest")
            ax[r, c].imshow(im, **kw); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0: ax[r, c].set_title(cols[c], fontsize=11)
        ax[r, 0].set_ylabel(f"{SUBTYPES[C[te][i]]}\npred: {SUBTYPES[cP[i].argmax()]}", fontsize=9)
        ax[r, 3].set_xlabel(f"mean Dice {res.Dice_P_vs_Path[i]:.3f}"); ax[r, 4].set_xlabel(f"mean Dice {res.Dice_AI_vs_Path[i]:.3f}")
    fig.legend(handles=[matplotlib.patches.Patch(color=cols_rgb[i], label=SEG_NAMES[i]) for i in range(NUM_SEG)],
               loc="lower center", ncol=9, fontsize=9)
    plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(f"{OUT}/qualitative.png", dpi=200); plt.close()
    print(f"Done. Everything is in ./{OUT}/")
