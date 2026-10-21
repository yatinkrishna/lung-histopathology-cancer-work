# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:08:57 2026

@author: Pooja
"""

"""
IGNITE H&E Multi-Task (Segmentation + Classification) Ablation Study
=====================================================================
Novel contributions targeted for the conference paper:
  (1) Class-imbalance-stratified fusion analysis: does attention/adaptive
      fusion actually help RARE tissue classes, or only the dominant ones?
  (2) Directional task-coupling ablation: does segmentation help
      classification, does classification help segmentation, or only
      mutual coupling helps? (multi_dir_seg_to_cls / multi_dir_cls_to_seg)

This is a CUSTOM PyTorch pipeline, independent of your existing nnU-Net
inference.py / evaluation.py (nnU-Net has no classification head, so a
joint model can't be expressed inside that framework).

Expected layout (adjust ROOT below if needed):
    ROOT\\Data\\images\\he\\*.png
    ROOT\\Data\\annotations\\he\\*.png        (17-class label-index masks)
    ROOT\\Data\\data_overview.csv             (manifest: task, split,
                                               histological_subtype, patient_id, ...)

Only rows with task == 'he_tissue_segmentation' are used.
Segmentation: 17 classes (0=Unannotated, 1=Background, 2..16 tissue types).
Classification: histological_subtype (e.g. adenocarcinoma / squamous_cell_carcinoma).

Rename this .txt to .py before running in Spyder.
"""

import os
import copy
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

ROOT = r"E:\ignite tool kit he segmentation\Data"
CSV_PATH = os.path.join(ROOT, "data_overview.csv")
IMG_DIR = os.path.join(ROOT, "images", "he")
MASK_DIR = os.path.join(ROOT, "annotations", "he")
OUTPUT_DIR = os.path.join(ROOT, "..", "multitask_ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 256
NUM_EPOCHS = 25          # smoke-test with 3-5 first; 8 variants x this = total runtime
BATCH_SIZE = 8
LR = 1e-3
VAL_PATIENT_FRACTION = 0.2
SEED = 42

# CV_MODE controls how train/val is split:
#   "single_split" - one fixed 80/20 patient split, 8 variants total trained once each.
#                     Use this for exploring which variants look promising.
#   "5fold"        - uses the manifest's own validation_fold column (0-4), trains every
#                     variant once per fold (8 x 5 = 40 runs) and reports mean +/- std
#                     per variant. Use this for the paper's final reported numbers, or
#                     narrow VARIANTS (below) down to just the winning ones first to
#                     keep runtime sane.
CV_MODE = "single_split"
NUM_WORKERS = 0          # Windows/Spyder safe default
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILM_DIM = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")

# Segmentation label names (from your evaluation.py) - class 0 excluded from loss/metrics
SEG_LABELS = {
    'Unannotated': 0, 'Background': 1, 'Tumor cells': 2, 'Reactive epithelium': 3,
    'Stroma': 4, 'Inflammation': 5, 'Alveolar tissue': 6, 'Fatty tissue': 7,
    'Necrotic tissue': 8, 'Erythrocytes': 9, 'Bronchial epithelium': 10,
    'Mucus/Plasma/Fluids': 11, 'Cartilage/Bone': 12, 'Macrophages': 13,
    'Muscle': 14, 'Liver': 15, 'Keratinization': 16,
}
NUM_SEG_CLASSES = 17
IDX_TO_LABEL = {v: k for k, v in SEG_LABELS.items()}
# Classes used for the stratified / paper-facing analysis (drop Unannotated + Background,
# matching your existing evaluation.py convention)
STRATIFY_CLASSES = [c for c in range(NUM_SEG_CLASSES) if c not in (0, 1)]

VARIANTS = [
    "seg_only",
    "cls_only",
    "multi_naive",
    "multi_attention",
    "multi_efficientnet",
    "multi_adaptive_weight",
    "multi_dir_seg_to_cls",
    "multi_dir_cls_to_seg",
]

# --------------------------------------------------------------------------
# MANIFEST / SPLIT
# --------------------------------------------------------------------------

def load_manifest():
    df = pd.read_csv(CSV_PATH)
    df = df[df["task"] == "he_tissue_segmentation"].copy()
    df["img_name"] = df["image_path"].apply(lambda p: os.path.basename(str(p)))
    df["mask_name"] = df["annotation_path"].apply(lambda p: os.path.basename(str(p)))
    df["img_full"] = df["img_name"].apply(lambda n: os.path.join(IMG_DIR, n))
    df["mask_full"] = df["mask_name"].apply(lambda n: os.path.join(MASK_DIR, n))
    exists = df["img_full"].apply(os.path.exists) & df["mask_full"].apply(os.path.exists)
    missing = (~exists).sum()
    if missing:
        print(f"[warning] {missing} manifest rows have no matching image+mask on disk, skipping them")
    df = df[exists].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No he_tissue_segmentation rows matched files on disk - check ROOT/paths.")
    print(f"Manifest: {len(df)} usable he_tissue_segmentation rows, "
          f"{df['patient_id'].nunique()} unique patients")
    return df


def build_subtype_mapping(df):
    subtypes = sorted(df["histological_subtype"].dropna().unique().tolist())
    subtype_to_idx = {s: i for i, s in enumerate(subtypes)}
    print(f"Classification classes ({len(subtypes)}): {subtypes}")
    return subtype_to_idx


def patient_level_split(df, val_fraction=VAL_PATIENT_FRACTION, seed=SEED):
    """Single fixed split by patient_id (not by ROI) to avoid leakage across train/val.
    Only uses split=='train' rows from the manifest as the training pool;
    the manifest's own 'test' rows are left untouched as a true held-out set."""
    train_pool = df[df["split"] == "train"]
    patients = sorted(train_pool["patient_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_fraction))
    val_patients = set(patients[:n_val])
    train_patients = set(patients[n_val:])
    print(f"Patient split: {len(train_patients)} train patients, {len(val_patients)} val patients")
    return train_patients, val_patients


def patient_fold_assignment(df):
    """Map each patient_id to a single validation_fold (0-4), using the majority
    fold value among that patient's rows (a patient's ROIs should all share one
    fold in a proper nnU-Net-style split, but this guards against stray mismatches)."""
    train_pool = df[df["split"] == "train"].dropna(subset=["validation_fold"])
    fold_of = {}
    for pid, group in train_pool.groupby("patient_id"):
        folds = group["validation_fold"].astype(str).str.replace("fold", "", regex=False)
        fold_of[pid] = int(folds.mode().iloc[0])
    return fold_of


def get_fold_split(df, fold_of, fold_idx):
    """train/val patient sets for one CV fold, using the manifest's own fold labels."""
    val_patients = {p for p, f in fold_of.items() if f == fold_idx}
    train_patients = {p for p, f in fold_of.items() if f != fold_idx}
    print(f"Fold {fold_idx}: {len(train_patients)} train patients, {len(val_patients)} val patients")
    return train_patients, val_patients


# --------------------------------------------------------------------------
# DATASET
# --------------------------------------------------------------------------

class IgniteMultiTaskDataset(Dataset):
    def __init__(self, df, patient_ids, subtype_to_idx, img_size=IMG_SIZE, augment=False):
        self.df = df[df["patient_id"].isin(patient_ids)].reset_index(drop=True)
        self.subtype_to_idx = subtype_to_idx
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_full"]).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(row["mask_full"]).convert("L").resize(
            (self.img_size, self.img_size), Image.NEAREST)
        mask_arr = np.array(mask).astype(np.int64)
        mask_arr = 255 - mask_arr  # masks are stored as (255 - class_index), see check_mask_encoding.py
        mask_arr = np.clip(mask_arr, 0, NUM_SEG_CLASSES - 1)  # safety

        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask_arr = np.fliplr(mask_arr).copy()
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask_arr = np.flipud(mask_arr).copy()
            if random.random() > 0.3:
                img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
                img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_t = torch.from_numpy(mask_arr)  # (H,W) long
        label = self.subtype_to_idx[row["histological_subtype"]]
        return img_t, mask_t, label


def compute_class_weights(train_df, subtype_to_idx, sample_cap=200):
    """One-time pass over (a sample of) the training masks to get inverse-frequency
    class weights for the segmentation CE loss."""
    counts = np.zeros(NUM_SEG_CLASSES, dtype=np.float64)
    sample_paths = train_df["mask_full"].tolist()
    if len(sample_paths) > sample_cap:
        sample_paths = random.sample(sample_paths, sample_cap)
    for p in sample_paths:
        arr = np.array(Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
        arr = 255 - arr.astype(np.int64)  # masks are stored as (255 - class_index)
        for c in range(NUM_SEG_CLASSES):
            counts[c] += (arr == c).sum()
    counts[0] = 0  # ignore Unannotated in weighting
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (counts * NUM_SEG_CLASSES)
    weights[0] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------
# ENCODERS
# --------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PlainEncoder(nn.Module):
    """5 stages -> skips at stride 2/4/8/16 (4 skips) + bottleneck at stride 32."""
    def __init__(self, base_ch=32):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.blocks = nn.ModuleList()
        in_ch = 3
        for ch in chs:
            self.blocks.append(nn.Sequential(DoubleConv(in_ch, ch), nn.MaxPool2d(2)))
            in_ch = ch
        self.skip_channels = chs[:-1]      # [32,64,128,256]
        self.bottleneck_channels = chs[-1]  # 512

    def forward(self, x):
        skips = []
        h = x
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i < len(self.blocks) - 1:
                skips.append(h)
        return skips, h  # skips: shallow->deep, h: bottleneck


class EfficientNetEncoder(nn.Module):
    """torchvision efficientnet_b0 sliced into the same shallow->deep skip pattern."""
    def __init__(self, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        eff = efficientnet_b0(weights=weights)
        feats = eff.features
        self.stage1 = nn.Sequential(*feats[0:2])   # stride2,  16ch
        self.stage2 = feats[2]                     # stride4,  24ch
        self.stage3 = feats[3]                     # stride8,  40ch
        self.stage4 = nn.Sequential(*feats[4:6])   # stride16, 112ch
        self.stage5 = nn.Sequential(*feats[6:9])   # stride32, 1280ch
        self.skip_channels = [16, 24, 40, 112]
        self.bottleneck_channels = 1280

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        bottleneck = self.stage5(s4)
        return [s1, s2, s3, s4], bottleneck


# --------------------------------------------------------------------------
# DECODER (dynamic: builds from whatever encoder's channel dims are)
# --------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, skip_channels, bottleneck_channels, num_seg_classes, film_dim=None):
        super().__init__()
        rev_skips = list(reversed(skip_channels))  # deepest skip first
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        prev_ch = bottleneck_channels
        for ch in rev_skips:
            self.ups.append(nn.ConvTranspose2d(prev_ch, ch, 2, stride=2))
            self.convs.append(DoubleConv(ch * 2, ch))
            prev_ch = ch
        self.final_up = nn.ConvTranspose2d(prev_ch, prev_ch // 2, 2, stride=2)
        self.final_conv = DoubleConv(prev_ch // 2, prev_ch // 2)
        self.out_conv = nn.Conv2d(prev_ch // 2, num_seg_classes, 1)

        self.film_dim = film_dim
        if film_dim is not None:
            self.film_fc = nn.Linear(film_dim, (prev_ch // 2) * 2)  # gamma+beta

    def forward(self, bottleneck, skips, film_vec=None, out_size=None):
        h = bottleneck
        for up, conv, skip in zip(self.ups, self.convs, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = conv(torch.cat([h, skip], dim=1))

        h = self.final_up(h)
        h = self.final_conv(h)

        if self.film_dim is not None and film_vec is not None:
            gb = self.film_fc(film_vec)
            gamma, beta = gb.chunk(2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + gamma) + beta

        decoder_final_feat = h  # exposed for seg->cls context injection
        if out_size is not None and h.shape[-2:] != out_size:
            h = F.interpolate(h, size=out_size, mode="bilinear", align_corners=False)
        seg_logits = self.out_conv(h)
        return seg_logits, decoder_final_feat


# --------------------------------------------------------------------------
# MULTI-TASK MODEL
# --------------------------------------------------------------------------

class MultiTaskModel(nn.Module):
    """
    variant controls architecture + information flow:
      seg_only               - encoder+decoder only
      cls_only                - encoder + cls head only (no decoder)
      multi_naive              - both heads, independent, no cross-injection
      multi_attention          - mutual: seg-context -> cls head, cls-embedding -> decoder (FiLM), both live
      multi_efficientnet       - multi_naive fusion, EfficientNet-B0 backbone instead of plain encoder
      multi_adaptive_weight    - multi_naive architecture; loss combination changes (handled outside model)
      multi_dir_seg_to_cls     - seg-context -> cls head (detached: seg loss unaffected by cls loss)
      multi_dir_cls_to_seg     - cls-embedding -> decoder FiLM (detached: cls loss unaffected by seg loss)
    """
    def __init__(self, variant, num_seg_classes, num_cls_classes):
        super().__init__()
        self.variant = variant
        backbone = "efficientnet" if variant == "multi_efficientnet" else "plain"
        self.encoder = EfficientNetEncoder() if backbone == "efficientnet" else PlainEncoder()

        self.has_seg = variant != "cls_only"
        self.has_cls = variant != "seg_only"
        self.uses_decoder_film = variant in ("multi_attention", "multi_dir_cls_to_seg")
        self.uses_seg_context_for_cls = variant in ("multi_attention", "multi_dir_seg_to_cls")
        self.detach_cls_to_seg = variant == "multi_dir_cls_to_seg"
        self.detach_seg_to_cls = variant == "multi_dir_seg_to_cls"

        film_dim = FILM_DIM if self.uses_decoder_film else None
        if self.has_seg:
            self.decoder = Decoder(self.encoder.skip_channels, self.encoder.bottleneck_channels,
                                    num_seg_classes, film_dim=film_dim)
        if self.uses_decoder_film:
            self.film_embed_fc = nn.Linear(self.encoder.bottleneck_channels, FILM_DIM)

        if self.has_cls:
            cls_in_dim = self.encoder.bottleneck_channels
            # figure out decoder feature channel count for context injection
            if self.uses_seg_context_for_cls:
                ctx_ch = self.decoder.final_conv.net[-3].out_channels  # channels of decoder_final_feat
                cls_in_dim += ctx_ch
            self.cls_head = nn.Sequential(
                nn.Linear(cls_in_dim, 128), nn.ReLU(inplace=True),
                nn.Dropout(0.3), nn.Linear(128, num_cls_classes),
            )

        if variant == "multi_adaptive_weight":
            self.log_var_seg = nn.Parameter(torch.zeros(1))
            self.log_var_cls = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        skips, bottleneck = self.encoder(x)
        seg_logits, decoder_final_feat = None, None
        cls_logits = None

        film_vec = None
        if self.uses_decoder_film:
            raw = self.film_embed_fc(F.adaptive_avg_pool2d(bottleneck, 1).flatten(1))
            film_vec = raw.detach() if self.detach_cls_to_seg else raw

        if self.has_seg:
            seg_logits, decoder_final_feat = self.decoder(bottleneck, skips, film_vec=film_vec,
                                                           out_size=x.shape[-2:])

        if self.has_cls:
            base_vec = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
            if self.uses_seg_context_for_cls and decoder_final_feat is not None:
                ctx = F.adaptive_avg_pool2d(decoder_final_feat, 1).flatten(1)
                ctx = ctx.detach() if self.detach_seg_to_cls else ctx
                cls_in = torch.cat([base_vec, ctx], dim=1)
            else:
                cls_in = base_vec
            cls_logits = self.cls_head(cls_in)

        return seg_logits, cls_logits


# --------------------------------------------------------------------------
# LOSSES
# --------------------------------------------------------------------------

class MultiClassDiceLoss(nn.Module):
    """Soft dice over classes 1..N-1 (ignores class 0 = Unannotated)."""
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target.clamp(0, self.num_classes - 1), self.num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        valid = (target != 0).unsqueeze(1).float()
        probs = probs[:, 1:] * valid
        target_onehot = target_onehot[:, 1:] * valid
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        union = probs.sum(dims) + target_onehot.sum(dims)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def seg_loss_fn(ce, dice_loss, logits, target):
    return ce(logits, target) + dice_loss(logits, target)


# --------------------------------------------------------------------------
# METRICS
# --------------------------------------------------------------------------

def dice_per_class_from_cm(cm):
    dice_scores = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        denom = 2 * tp + fp + fn
        dice_scores.append(2 * tp / denom if denom > 0 else np.nan)
    return dice_scores


# --------------------------------------------------------------------------
# TRAIN / EVAL ONE VARIANT
# --------------------------------------------------------------------------

def train_one_variant(variant, train_loader, val_loader, class_weights, num_cls_classes,
                       num_epochs=NUM_EPOCHS, verbose=True):
    model = MultiTaskModel(variant, NUM_SEG_CLASSES, num_cls_classes).to(DEVICE)
    ce = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), ignore_index=0)
    dice_loss = MultiClassDiceLoss(NUM_SEG_CLASSES)
    cls_ce = nn.CrossEntropyLoss()

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    best_score = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks, labels in train_loader:
            imgs, masks, labels = imgs.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            seg_logits, cls_logits = model(imgs)

            loss = 0.0
            seg_l = seg_loss_fn(ce, dice_loss, seg_logits, masks) if seg_logits is not None else None
            cls_l = cls_ce(cls_logits, labels) if cls_logits is not None else None

            if variant == "seg_only":
                loss = seg_l
            elif variant == "cls_only":
                loss = cls_l
            elif variant == "multi_adaptive_weight":
                precision_seg = torch.exp(-model.log_var_seg)
                precision_cls = torch.exp(-model.log_var_cls)
                loss = (precision_seg * seg_l + model.log_var_seg +
                        precision_cls * cls_l + model.log_var_cls).squeeze()
            else:
                loss = seg_l + cls_l

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        running_loss /= len(train_loader.dataset)

        # --- validation ---
        model.eval()
        cm = np.zeros((NUM_SEG_CLASSES, NUM_SEG_CLASSES), dtype=np.int64) if model.has_seg else None
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, masks, labels in val_loader:
                imgs = imgs.to(DEVICE)
                seg_logits, cls_logits = model(imgs)
                if seg_logits is not None:
                    preds = seg_logits.argmax(dim=1).cpu().numpy().reshape(-1)
                    gt = masks.numpy().reshape(-1)
                    cm += confusion_matrix(gt, preds, labels=list(range(NUM_SEG_CLASSES)))
                if cls_logits is not None:
                    all_preds.extend(cls_logits.argmax(dim=1).cpu().numpy().tolist())
                    all_labels.extend(labels.numpy().tolist())

        mean_dice = None
        if model.has_seg:
            dice_scores = dice_per_class_from_cm(cm)
            mean_dice = np.nanmean([dice_scores[c] for c in STRATIFY_CLASSES])
        cls_acc = accuracy_score(all_labels, all_preds) if model.has_cls else None

        # model-selection score: mean_dice if seg present, else cls_acc
        score = mean_dice if mean_dice is not None else cls_acc
        scheduler.step(score)
        if score is not None and score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_cm = cm.copy() if cm is not None else None
            best_labels_preds = (list(all_labels), list(all_preds)) if model.has_cls else None

        if verbose:
            msg = f"  [{variant}] epoch {epoch:02d}/{num_epochs} loss={running_loss:.4f}"
            if mean_dice is not None:
                msg += f" val_mean_dice={mean_dice:.4f}"
            if cls_acc is not None:
                msg += f" val_cls_acc={cls_acc:.4f}"
            print(msg)

    result = {"variant": variant, "best_score": best_score}
    if model.has_seg:
        dice_scores = dice_per_class_from_cm(best_cm)
        result["per_class_dice"] = {IDX_TO_LABEL[c]: dice_scores[c] for c in range(NUM_SEG_CLASSES)}
        result["mean_dice"] = np.nanmean([dice_scores[c] for c in STRATIFY_CLASSES])
    if model.has_cls:
        labs, preds = best_labels_preds
        result["cls_accuracy"] = accuracy_score(labs, preds)
        result["cls_macro_f1"] = f1_score(labs, preds, average="macro")

    return result, best_state


# --------------------------------------------------------------------------
# ABLATION RUNNER + ANALYSES
# --------------------------------------------------------------------------

def run_one_split(df, subtype_to_idx, train_patients, val_patients, label="single_split"):
    num_cls_classes = len(subtype_to_idx)
    train_df = df[df["patient_id"].isin(train_patients) & (df["split"] == "train")]
    class_weights = compute_class_weights(train_df, subtype_to_idx)
    print("Segmentation class weights (index 0..16):", class_weights.tolist())

    train_ds = IgniteMultiTaskDataset(df, train_patients, subtype_to_idx, augment=True)
    val_ds = IgniteMultiTaskDataset(df, val_patients, subtype_to_idx, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # class frequency tiers, computed once from the training set, for the stratified analysis
    freq = np.zeros(NUM_SEG_CLASSES)
    for p in random.sample(train_df["mask_full"].tolist(), min(150, len(train_df))):
        arr = np.array(Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
        arr = 255 - arr.astype(np.int64)  # masks are stored as (255 - class_index)
        for c in STRATIFY_CLASSES:
            freq[c] += (arr == c).sum()
    ranked = sorted(STRATIFY_CLASSES, key=lambda c: freq[c])
    n = len(ranked)
    rare_tier = ranked[: n // 3]
    mid_tier = ranked[n // 3: 2 * n // 3]
    common_tier = ranked[2 * n // 3:]
    print("Rare classes:", [IDX_TO_LABEL[c] for c in rare_tier])
    print("Mid classes:", [IDX_TO_LABEL[c] for c in mid_tier])
    print("Common classes:", [IDX_TO_LABEL[c] for c in common_tier])

    all_results = []
    for i, variant in enumerate(VARIANTS, 1):
        print(f"\n=== [{i}/{len(VARIANTS)}] {variant} ===")
        t0 = time.time()
        try:
            result, state = train_one_variant(variant, train_loader, val_loader, class_weights, num_cls_classes)
        except Exception as e:
            print(f"    FAILED: {e}")
            all_results.append({"variant": variant, "error": str(e)})
            continue
        elapsed = time.time() - t0
        result["time_sec"] = elapsed
        torch.save(state, os.path.join(OUTPUT_DIR, f"{label}_{variant}_best.pt"))
        all_results.append(result)
        print(f"    done in {elapsed/60:.1f} min - {result}")

    # ---- summary table ----
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "variant": r.get("variant"),
            "mean_dice": r.get("mean_dice"),
            "cls_accuracy": r.get("cls_accuracy"),
            "cls_macro_f1": r.get("cls_macro_f1"),
            "time_sec": r.get("time_sec"),
            "error": r.get("error"),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, f"summary_{label}.csv"), index=False)
    print(f"\n================ SUMMARY ({label}) ================")
    print(summary_df.to_string(index=False))

    # ---- contribution 1: class-imbalance-stratified analysis ----
    strat_rows = []
    for r in all_results:
        if "per_class_dice" not in r:
            continue
        pcd = r["per_class_dice"]
        strat_rows.append({
            "variant": r["variant"],
            "rare_mean_dice": np.nanmean([pcd[IDX_TO_LABEL[c]] for c in rare_tier]),
            "mid_mean_dice": np.nanmean([pcd[IDX_TO_LABEL[c]] for c in mid_tier]),
            "common_mean_dice": np.nanmean([pcd[IDX_TO_LABEL[c]] for c in common_tier]),
        })
    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(os.path.join(OUTPUT_DIR, f"class_imbalance_stratified_{label}.csv"), index=False)
    print("\n======== CLASS-IMBALANCE-STRATIFIED (contribution 1) ========")
    print(strat_df.to_string(index=False))

    # ---- contribution 2: directional task-coupling analysis ----
    dir_variants = ["seg_only", "cls_only", "multi_naive", "multi_attention",
                     "multi_dir_seg_to_cls", "multi_dir_cls_to_seg"]
    dir_df = summary_df[summary_df["variant"].isin(dir_variants)].copy()
    dir_df.to_csv(os.path.join(OUTPUT_DIR, f"directional_ablation_{label}.csv"), index=False)
    print("\n======== DIRECTIONAL TASK-COUPLING (contribution 2) ========")
    print(dir_df.to_string(index=False))
    print(f"\nResults for {label} saved under: {OUTPUT_DIR}")

    return summary_df, strat_df, dir_df


def run_ablation():
    """Dispatcher: runs either one fixed split or full 5-fold CV, per CV_MODE,
    and (for 5fold) aggregates each variant's metrics into mean +/- std."""
    df = load_manifest()
    subtype_to_idx = build_subtype_mapping(df)

    if CV_MODE == "single_split":
        train_patients, val_patients = patient_level_split(df)
        return run_one_split(df, subtype_to_idx, train_patients, val_patients, label="single_split")

    elif CV_MODE == "5fold":
        fold_of = patient_fold_assignment(df)
        available_folds = sorted(set(fold_of.values()))
        print(f"Running full CV over folds: {available_folds} "
              f"({len(VARIANTS)} variants x {len(available_folds)} folds = "
              f"{len(VARIANTS) * len(available_folds)} training runs)")

        per_fold_summaries = []
        for fold_idx in available_folds:
            train_patients, val_patients = get_fold_split(df, fold_of, fold_idx)
            summary_df, _, _ = run_one_split(df, subtype_to_idx, train_patients, val_patients,
                                              label=f"fold{fold_idx}")
            summary_df["fold"] = fold_idx
            per_fold_summaries.append(summary_df)

        all_folds_df = pd.concat(per_fold_summaries, ignore_index=True)
        all_folds_df.to_csv(os.path.join(OUTPUT_DIR, "summary_all_folds.csv"), index=False)

        agg = (all_folds_df.groupby("variant")[["mean_dice", "cls_accuracy", "cls_macro_f1"]]
               .agg(["mean", "std"]))
        agg.columns = ["_".join(c) for c in agg.columns]
        agg = agg.reset_index()
        agg.to_csv(os.path.join(OUTPUT_DIR, "summary_5fold_aggregated.csv"), index=False)
        print("\n================ 5-FOLD AGGREGATED SUMMARY (mean +/- std) ================")
        print(agg.to_string(index=False))
        print(f"\nAll per-fold and aggregated results saved under: {OUTPUT_DIR}")
        return agg

    else:
        raise ValueError(f"Unknown CV_MODE: {CV_MODE!r} - use 'single_split' or '5fold'")


if __name__ == "__main__":
    # Smoke test first: set NUM_EPOCHS = 3 above and CV_MODE = "single_split",
    # confirm this runs end to end and metrics move, before committing to the
    # full run (and before ever switching CV_MODE to "5fold").
    run_ablation()