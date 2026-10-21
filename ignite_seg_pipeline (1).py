"""
IGNITE H&E  |  BASTNet  (Boundary-aware, Attention-gated, Stain-invariant, Transformer hybrid)
=====================================================================================
One script does everything:
  1. Patient-level 70/20/10 split (train/val/test, no patient leakage)
  2. "AI Pathologist" = out-of-fold K-fold ensemble trained on pathologist masks
        -> produces AI ground truth (OOF for train, ensemble for val/test)
  3. Model-P  : trained on pathologist annotation
     Model-AI : trained on AI ground truth
  4. Joint segmentation + boundary + image-level classification (multi-task)
  5. Statistical patient-grouped K-fold CV, Wilcoxon signed-rank, IQR/box/violin plots
  6. t-SNE of learned features, ROC / confusion matrix for classification
  7. Qualitative figure:  Original | Pathologist GT | AI GT | Pred (Path-trained) | Pred (AI-trained) | Uncertainty

pip install torch torchvision timm opencv-python numpy pandas scikit-learn scipy matplotlib seaborn tqdm
"""
import os, re, glob, random, json, warnings
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, timm
from tqdm import tqdm
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, f1_score, accuracy_score)
warnings.filterwarnings("ignore")

# ============================== CONFIG ==============================
IMG_DIR      = "E:\ignite tool kit he segmentation\Data\organised_he/images"      # H&E images  (png/jpg/tif)
MASK_DIR     = "E:\ignite tool kit he segmentation\Data\organised_he/masks"       # annotations, same file stem as image
NUM_CLASSES  = 2                  # incl. background (0). Binary 0/255 masks are auto-converted
CLASS_NAMES  = ["Background", "Tumour"]          # edit to match NUM_CLASSES
PATIENT_REGEX = r"^([^_\.\-]+)"   # patient ID = first token of filename (e.g. P001_tile3.png -> P001)
IMG_SIZE     = 256                # multiple of 32
BATCH        = 16
EPOCHS       = 60                 # final models
EPOCHS_CV    = 25                 # K-fold / AI-pathologist models
K_FOLDS      = 5
LR           = 2e-4
PRETRAINED   = True               # timm ConvNeXt-Tiny ImageNet weights (needs internet once)
SEED         = 42
OUT          = "results"
DEV          = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)

# ============================== DATA ==============================
def load_all():
    exts = ("png", "jpg", "jpeg", "tif", "tiff", "bmp")
    imgs = sorted(sum([glob.glob(f"{IMG_DIR}/**/*.{e}", recursive=True) for e in exts], []))
    mk = {os.path.splitext(os.path.basename(p))[0]: p for e in exts
          for p in glob.glob(f"{MASK_DIR}/**/*.{e}", recursive=True)}
    X, Y, P, N = [], [], [], []
    for p in tqdm(imgs, desc="Loading"):
        stem = os.path.splitext(os.path.basename(p))[0]
        m = mk.get(stem) or mk.get(stem + "_mask") or mk.get(stem.replace("_img", "_mask"))
        if m is None: continue
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        ms = cv2.imread(m, cv2.IMREAD_UNCHANGED)
        if ms.ndim == 3: ms = ms[..., 0]
        im = cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        ms = cv2.resize(ms, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        if NUM_CLASSES == 2 and ms.max() > 1: ms = (ms > 127).astype(np.uint8) if ms.max() > 200 else (ms > 0).astype(np.uint8)
        ms = np.clip(ms, 0, NUM_CLASSES - 1).astype(np.uint8)
        X.append(im); Y.append(ms); N.append(stem)
        P.append(re.match(PATIENT_REGEX, stem).group(1))
    assert len(X) > 0, "No image/mask pairs found - check IMG_DIR, MASK_DIR"
    return np.stack(X), np.stack(Y), np.array(P), np.array(N)

def image_label(mask, thr=0.01):
    """Image-level class = dominant foreground class (0 if foreground < 1% of pixels)."""
    cnt = np.bincount(mask.ravel(), minlength=NUM_CLASSES)[1:]
    return 0 if cnt.sum() < thr * mask.size else int(cnt.argmax() + 1)

def stain_jitter(img, s=0.25):
    """Random perturbation in optical-density space (mimics stain/scanner variability)."""
    od = -np.log((img.astype(np.float32) + 1) / 256.)
    od = od * np.random.uniform(1 - s, 1 + s, (1, 1, 3)) + np.random.uniform(-s / 5, s / 5, (1, 1, 3))
    return np.clip(256 * np.exp(-od) - 1, 0, 255).astype(np.uint8)

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
def to_tensor(img): return torch.from_numpy(((img / 255. - MEAN) / STD).transpose(2, 0, 1).astype(np.float32))

class DS(torch.utils.data.Dataset):
    def __init__(self, X, Y, C, train): self.X, self.Y, self.C, self.t = X, Y, C, train
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        im, ms = self.X[i], self.Y[i]
        if self.t:
            k = np.random.randint(4); im, ms = np.rot90(im, k), np.rot90(ms, k)
            if np.random.rand() < .5: im, ms = im[:, ::-1], ms[:, ::-1]
            im = stain_jitter(np.ascontiguousarray(im), .15)
        im, ms = np.ascontiguousarray(im), np.ascontiguousarray(ms)
        v2 = stain_jitter(im, .35) if self.t else im         # second stain view for consistency loss
        return to_tensor(im), to_tensor(v2), torch.from_numpy(ms.astype(np.int64)), int(self.C[i])

# ============================== NOVEL MODEL ==============================
class ConvBR(nn.Sequential):
    def __init__(s, i, o, k=3, d=1): super().__init__(nn.Conv2d(i, o, k, padding=d * (k // 2), dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

class ASPP(nn.Module):                                       # multi-scale context (tissue architecture varies in scale)
    def __init__(s, i, o):
        super().__init__()
        s.b = nn.ModuleList([ConvBR(i, o, 1), ConvBR(i, o, 3, 2), ConvBR(i, o, 3, 4)])
        s.g = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(i, o, 1), nn.ReLU(True))
        s.p = ConvBR(o * 4, o, 1)
    def forward(s, x):
        g = s.g(x).expand(-1, -1, x.shape[2], x.shape[3])
        return s.p(torch.cat([b(x) for b in s.b] + [g], 1))

class GlobalTransformer(nn.Module):                          # long-range context at bottleneck
    def __init__(s, d, L=2, heads=8):
        super().__init__()
        s.pos = nn.Conv2d(d, d, 3, padding=1, groups=d)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, d * 2, 0.1, batch_first=True, norm_first=True), L)
    def forward(s, x):
        x = x + s.pos(x); B, C, H, W = x.shape
        return s.enc(x.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)

class AttGate(nn.Module):                                    # attention-gated skip connections
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
    """Encoder: ConvNeXt-T | Bottleneck: ASPP + Transformer | Decoder: attention-gated U-Net
       Heads: segmentation, boundary (edge-aware), image-level classification."""
    def __init__(s, nc=NUM_CLASSES, pretrained=PRETRAINED):
        super().__init__()
        s.enc = timm.create_model("convnext_tiny", pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        ch = s.enc.feature_info.channels()                   # 96,192,384,768
        s.aspp = ASPP(ch[3], 256); s.trf = GlobalTransformer(256)
        s.u3 = Up(256, ch[2], 192); s.u2 = Up(192, ch[1], 128); s.u1 = Up(128, ch[0], 64)
        s.seg = nn.Sequential(ConvBR(64, 32), nn.Conv2d(32, nc, 1))
        s.bnd = nn.Sequential(ConvBR(64, 32), nn.Conv2d(32, 1, 1))
        s.cls = nn.Sequential(nn.Linear(256 + 64, 128), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(128, nc))
    def forward(s, x):
        H, W = x.shape[2:]
        f = s.enc(x); b = s.trf(s.aspp(f[3]))
        d = s.u1(s.u2(s.u3(b, f[2]), f[1]), f[0])
        up = lambda t: F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        feat = torch.cat([b.mean((2, 3)), d.mean((2, 3))], 1)
        return up(s.seg(d)), up(s.bnd(d)), s.cls(feat), b.mean((2, 3))

def boundary_target(m):
    m = m.float().unsqueeze(1)
    return ((F.max_pool2d(m, 5, 1, 2) - (-F.max_pool2d(-m, 5, 1, 2))) > 0).float()

def dice_loss(lg, y):
    p = lg.softmax(1); oh = F.one_hot(y, NUM_CLASSES).permute(0, 3, 1, 2).float()
    i = (p * oh).sum((0, 2, 3)); u = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * i + 1) / (u + 1)).mean()

def focal_ce(lg, y, g=2.0):
    ce = F.cross_entropy(lg, y, reduction="none"); return (((1 - torch.exp(-ce)) ** g) * ce).mean()

# ============================== TRAIN / PREDICT ==============================
@torch.no_grad()
def predict(models, X, tta=True, bs=32, feats=False):
    """Ensemble (list of models) + flip TTA. Returns softmax probs, class probs, (features)."""
    models = models if isinstance(models, list) else [models]
    for m in models: m.eval()
    PS, CS, FS = [], [], []
    for i in range(0, len(X), bs):
        x = torch.stack([to_tensor(a) for a in X[i:i + bs]]).to(DEV)
        ps, cs, fs, n = 0, 0, 0, 0
        for m in models:
            for fl in ([[], [3], [2]] if tta else [[]]):
                xx = x.flip(fl) if fl else x
                s, _, c, f = m(xx)
                s = s.softmax(1); s = s.flip(fl) if fl else s
                ps, cs, fs, n = ps + s, cs + c.softmax(1), fs + f, n + 1
        PS.append((ps / n).cpu()); CS.append((cs / n).cpu()); FS.append((fs / n).cpu())
    out = (torch.cat(PS).numpy(), torch.cat(CS).numpy())
    return out + (torch.cat(FS).numpy(),) if feats else out

@torch.no_grad()
def mc_uncertainty(model, X, T=8, bs=16):
    """MC-dropout predictive entropy."""
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
    rows = []
    for p, g in zip(pred, gt):
        d, j = [], []
        for c in range(1, NUM_CLASSES):
            a, b = p == c, g == c
            if a.sum() + b.sum() == 0: continue
            inter = (a & b).sum(); d.append(2 * inter / (a.sum() + b.sum())); j.append(inter / ((a | b).sum()))
        rows.append(dict(Dice=np.mean(d) if d else 1.0, IoU=np.mean(j) if j else 1.0, PixAcc=(p == g).mean()))
    return pd.DataFrame(rows)

def train(Xtr, Ytr, Xva, Yva, epochs, tag):
    Ctr = [image_label(m) for m in Ytr]
    dl = torch.utils.data.DataLoader(DS(Xtr, Ytr, Ctr, True), BATCH, shuffle=True, num_workers=2, drop_last=len(Xtr) > BATCH)
    model = BASTNet().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=epochs * len(dl))
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda"); best, path = -1, f"{OUT}/{tag}.pt"
    for ep in range(epochs):
        model.train(); tl = 0
        for x, x2, y, c in dl:
            x, x2, y, c = x.to(DEV), x2.to(DEV), y.to(DEV), c.to(DEV)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                s, b, cl, f1 = model(x); _, _, _, f2 = model(x2)
                loss = (focal_ce(s, y) + dice_loss(s, y)
                        + 0.5 * F.binary_cross_entropy_with_logits(b, boundary_target(y))
                        + 0.3 * F.cross_entropy(cl, c)
                        + 0.1 * F.mse_loss(F.normalize(f1, dim=1), F.normalize(f2, dim=1)))   # stain-invariance
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); tl += loss.item()
        if len(Xva):
            pr, _ = predict(model, Xva, tta=False); d = per_image_metrics(pr.argmax(1), Yva).Dice.mean()
            if d > best: best = d; torch.save(model.state_dict(), path)
            print(f"[{tag}] ep{ep + 1}/{epochs} loss {tl / len(dl):.4f} val Dice {d:.4f} (best {best:.4f})")
        else: torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load(path)); return model.eval()

# ============================== MAIN ==============================
if __name__ == "__main__":
    X, Y, P, N = load_all(); C = np.array([image_label(m) for m in Y])
    print(f"{len(X)} images, {len(set(P))} patients")

    # ---- 70/20/10 patient-level split ----
    g1 = GroupShuffleSplit(1, test_size=0.10, random_state=SEED)
    tv, te = next(g1.split(X, groups=P))
    g2 = GroupShuffleSplit(1, test_size=2 / 9, random_state=SEED)      # 20% of total
    tr, va = next(g2.split(X[tv], groups=P[tv])); tr, va = tv[tr], tv[va]
    assert not (set(P[tr]) & set(P[va]) | set(P[tr]) & set(P[te]) | set(P[va]) & set(P[te]))
    pd.DataFrame({"name": N, "patient": P, "split": np.select([np.isin(np.arange(len(X)), tr), np.isin(np.arange(len(X)), va)], ["train", "val"], "test")}).to_csv(f"{OUT}/split.csv", index=False)
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} images")

    # ---- Patient-grouped K-fold CV  +  AI PATHOLOGIST (out-of-fold ensemble) ----
    AI_Y = np.zeros_like(Y); fold_models, cv_rows = [], []
    for k, (a, b) in enumerate(GroupKFold(K_FOLDS).split(tr, groups=P[tr])):
        m = train(X[tr[a]], Y[tr[a]], X[tr[b]], Y[tr[b]], EPOCHS_CV, f"cv_fold{k}")
        pr, cp = predict(m, X[tr[b]]); AI_Y[tr[b]] = pr.argmax(1)               # OOF = honest AI annotation
        d = per_image_metrics(pr.argmax(1), Y[tr[b]]); d["fold"] = k + 1
        d["cls_acc"] = accuracy_score(C[tr[b]], cp.argmax(1)); cv_rows.append(d); fold_models.append(m)
    cv = pd.concat(cv_rows); cv.to_csv(f"{OUT}/cv_metrics.csv", index=False)
    print(cv.groupby("fold")[["Dice", "IoU", "PixAcc", "cls_acc"]].mean().agg(["mean", "std"]).round(4))
    for idx in (va, te): AI_Y[idx] = predict(fold_models, X[idx])[0].argmax(1)   # ensemble = "AI pathologist" consensus
    np.save(f"{OUT}/ai_groundtruth.npy", AI_Y)

    # ---- Final models ----
    model_P  = train(X[tr], Y[tr],    X[va], Y[va],    EPOCHS, "model_pathologist")
    model_AI = train(X[tr], AI_Y[tr], X[va], AI_Y[va], EPOCHS, "model_AI")

    # ---- Test evaluation ----
    pP, cP, fP = predict(model_P, X[te], feats=True); pA, cA, fA = predict(model_AI, X[te], feats=True)
    prP, prA = pP.argmax(1), pA.argmax(1)
    mP  = per_image_metrics(prP, Y[te]);    mA  = per_image_metrics(prA, Y[te])      # both vs pathologist
    mPa = per_image_metrics(prP, AI_Y[te]); mAa = per_image_metrics(prA, AI_Y[te])   # both vs AI GT
    mAI_vs_path = per_image_metrics(AI_Y[te], Y[te])                                  # AI pathologist vs human
    res = pd.DataFrame({"image": N[te], "patient": P[te],
        "Dice_P_vs_Path": mP.Dice, "Dice_AI_vs_Path": mA.Dice, "IoU_P_vs_Path": mP.IoU, "IoU_AI_vs_Path": mA.IoU,
        "Dice_P_vs_AI": mPa.Dice, "Dice_AI_vs_AI": mAa.Dice, "Dice_AIpath_vs_Path": mAI_vs_path.Dice})
    res.to_csv(f"{OUT}/test_per_image.csv", index=False)

    # ---- Statistics (Wilcoxon signed-rank, paired; IQR) ----
    stats = []
    for name, a, b in [("Dice: Model-P vs Model-AI (vs pathologist GT)", res.Dice_P_vs_Path, res.Dice_AI_vs_Path),
                       ("IoU : Model-P vs Model-AI (vs pathologist GT)", res.IoU_P_vs_Path, res.IoU_AI_vs_Path),
                       ("Dice: Model-P vs AI-pathologist mask (vs human GT)", res.Dice_P_vs_Path, res.Dice_AIpath_vs_Path)]:
        try: w, p = wilcoxon(a, b)
        except ValueError: w, p = np.nan, np.nan
        q = lambda s: (s.quantile(.25), s.median(), s.quantile(.75))
        stats.append(dict(test=name, W=w, p=p, A_Q1_med_Q3=np.round(q(a), 4).tolist(), B_Q1_med_Q3=np.round(q(b), 4).tolist()))
    pd.DataFrame(stats).to_csv(f"{OUT}/wilcoxon_stats.csv", index=False); print(pd.DataFrame(stats).to_string())

    # ---- Classification ----
    cls_out = {}
    for tag, cp in (("Model-P", cP), ("Model-AI", cA)):
        yt, yp = C[te], cp.argmax(1)
        cls_out[tag] = dict(acc=accuracy_score(yt, yp), f1_macro=f1_score(yt, yp, average="macro"))
        try: cls_out[tag]["auc_ovr"] = roc_auc_score(yt, cp if NUM_CLASSES > 2 else cp[:, 1], multi_class="ovr")
        except Exception: pass
        print(tag, classification_report(yt, yp, zero_division=0))
    json.dump(cls_out, open(f"{OUT}/classification_metrics.json", "w"), indent=2)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, (tag, cp) in enumerate((("Model-P", cP), ("Model-AI", cA))):
        sns.heatmap(confusion_matrix(C[te], cp.argmax(1), labels=range(NUM_CLASSES)), annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax[i]); ax[i].set_title(f"Classification - {tag}")
    for tag, cp in (("Model-P", cP), ("Model-AI", cA)):
        if NUM_CLASSES == 2 and len(set(C[te])) == 2:
            fpr, tpr, _ = roc_curve(C[te], cp[:, 1]); ax[2].plot(fpr, tpr, label=f"{tag} AUC={roc_auc_score(C[te], cp[:, 1]):.3f}")
    ax[2].plot([0, 1], [0, 1], "k--"); ax[2].legend(); ax[2].set_title("ROC"); plt.tight_layout(); plt.savefig(f"{OUT}/classification.png", dpi=200); plt.close()

    # ---- IQR box / violin / paired plots ----
    long = res.melt(id_vars=["image"], value_vars=["Dice_P_vs_Path", "Dice_AI_vs_Path", "Dice_AIpath_vs_Path"], var_name="Model", value_name="Dice")
    long["Model"] = long.Model.map({"Dice_P_vs_Path": "Model-P", "Dice_AI_vs_Path": "Model-AI", "Dice_AIpath_vs_Path": "AI-pathologist mask"})
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    sns.boxplot(data=long, x="Model", y="Dice", ax=ax[0], showmeans=True, whis=1.5); sns.stripplot(data=long, x="Model", y="Dice", ax=ax[0], color="k", size=2, alpha=.4)
    ax[0].set_title("Test Dice - median & IQR (box), 1.5xIQR whiskers"); sns.violinplot(data=long, x="Model", y="Dice", inner="quartile", ax=ax[1]); ax[1].set_title("Distribution (quartiles)")
    ax[2].scatter(res.Dice_P_vs_Path, res.Dice_AI_vs_Path, s=14); ax[2].plot([0, 1], [0, 1], "r--")
    ax[2].set_xlabel("Dice Model-P"); ax[2].set_ylabel("Dice Model-AI"); ax[2].set_title(f"Paired (Wilcoxon p={stats[0]['p']:.3g})")
    plt.tight_layout(); plt.savefig(f"{OUT}/iqr_wilcoxon_plots.png", dpi=200); plt.close()
    plt.figure(figsize=(7, 4.5)); sns.boxplot(data=cv, x="fold", y="Dice"); plt.title(f"Patient-grouped {K_FOLDS}-fold CV (Dice, IQR)")
    plt.tight_layout(); plt.savefig(f"{OUT}/cv_boxplot.png", dpi=200); plt.close()

    # ---- t-SNE of bottleneck features (val+test) ----
    idx = np.concatenate([va, te]); _, _, fv = predict(model_P, X[idx], feats=True); _, _, fva = predict(model_AI, X[idx], feats=True)
    split_lab = np.array(["val"] * len(va) + ["test"] * len(te))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.3))
    for i, (f, t) in enumerate(((fv, "Model-P"), (fva, "Model-AI"))):
        z = PCA(min(50, len(f) - 1)).fit_transform(f); z = TSNE(2, perplexity=min(30, max(5, len(f) // 4)), random_state=SEED, init="pca").fit_transform(z)
        sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=[CLASS_NAMES[c] for c in C[idx]], style=split_lab, ax=ax[i], s=35); ax[i].set_title(f"t-SNE features - {t}")
        if i == 0: zz = z
    sns.scatterplot(x=zz[:, 0], y=zz[:, 1], hue=P[idx], legend=False, ax=ax[2], s=30, palette="tab20"); ax[2].set_title("t-SNE Model-P coloured by patient (batch-effect check)")
    plt.tight_layout(); plt.savefig(f"{OUT}/tsne.png", dpi=200); plt.close()

    # ---- Qualitative figure ----
    cmap = matplotlib.colors.ListedColormap(plt.cm.tab10(np.arange(NUM_CLASSES)))
    unc = mc_uncertainty(model_P, X[te][:8]); show = np.argsort(-res.Dice_P_vs_Path.values)[:2].tolist() + np.argsort(res.Dice_P_vs_Path.values)[:2].tolist()
    show = list(dict.fromkeys(show + list(range(min(4, len(te))))))[:6]; U = mc_uncertainty(model_P, X[te][show])
    cols = ["Original H&E", "Pathologist GT", "AI-pathologist GT", "Pred (trained on pathologist)", "Pred (trained on AI GT)", "Uncertainty (MC-dropout)"]
    fig, ax = plt.subplots(len(show), 6, figsize=(21, 3.5 * len(show)))
    for r, i in enumerate(show):
        ims = [X[te][i], Y[te][i], AI_Y[te][i], prP[i], prA[i], U[r]]
        for c, im in enumerate(ims):
            ax[r, c].imshow(im, **({} if c == 0 else dict(cmap="magma") if c == 5 else dict(cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest"))); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(cols[c], fontsize=11)
        ax[r, 3].set_xlabel(f"Dice {res.Dice_P_vs_Path[i]:.3f}"); ax[r, 4].set_xlabel(f"Dice {res.Dice_AI_vs_Path[i]:.3f}")
    plt.tight_layout(); plt.savefig(f"{OUT}/qualitative.png", dpi=200); plt.close()
    print(f"Done. Everything is in ./{OUT}/")
