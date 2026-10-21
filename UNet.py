# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:42:44 2026

@author: Pooja
"""

# ============================================================
# CONFIG – edit these lines
# ============================================================
# ============================================================
# CONFIG
# ============================================================
DATA_ROOT      = r"E:\ignite tool kit he segmentation\Data\organised_he"
CSV_PATH       = r"E:\ignite tool kit he segmentation\Data\data_overview.csv"
OUTPUT_DIR     = r"E:\ignite tool kit he segmentation\Data\unet_results"

ORGAN          = "lung"
NUM_CLASSES    = 2
BATCH_SIZE     = 16          # good starting point for 16–24 GB VRAM
NUM_EPOCHS     = 50
LR             = 1e-4
IMG_SIZE       = 512
NUM_WORKERS    = 0           # safe on Windows with recent PyTorch
SEED           = 42
DEVICE         = "cuda"
# ============================================================
# ============================================================

import os, json, random, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# optional but recommended
import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

OUTPUT = Path(OUTPUT_DIR)
OUTPUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1. Dataset
# ------------------------------------------------------------------
class HESegDataset(Dataset):
    def __init__(self, df, data_root, transform=None, img_size=512):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subtype = str(row["histological_subtype"]).strip().lower().replace(" ", "_")
        if not subtype or subtype == "nan":
            subtype = "unknown"

        stem = Path(row["name"]).stem
        img_path  = self.data_root / subtype / "images" / f"{stem}.png"
        mask_path = self.data_root / subtype / "masks"  / f"{stem}.png"

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path))

        # binarise if needed (background=0, everything else=1)
        if NUM_CLASSES == 2:
            mask = (mask > 0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]
        else:
            image = torch.from_numpy(image.transpose(2,0,1)).float() / 255.
            mask  = torch.from_numpy(mask).long()

        return image, mask.long()


# ------------------------------------------------------------------
# 2. Augmentations
# ------------------------------------------------------------------
train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])


# ------------------------------------------------------------------
# 3. Metrics (vectorised, publication-ready)
# ------------------------------------------------------------------
def compute_metrics(pred, target, num_classes=2, ignore_index=None):
    """
    pred, target : (B,H,W) long tensors
    Returns dict of mean metrics over the batch
    """
    pred = pred.cpu().numpy().ravel()
    target = target.cpu().numpy().ravel()

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]

    metrics = {}
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        tp = np.logical_and(p, t).sum()
        fp = np.logical_and(p, ~t).sum()
        fn = np.logical_and(~p, t).sum()
        tn = np.logical_and(~p, ~t).sum()

        dice  = (2*tp) / (2*tp + fp + fn + 1e-8)
        iou   = tp / (tp + fp + fn + 1e-8)
        prec  = tp / (tp + fp + 1e-8)
        rec   = tp / (tp + fn + 1e-8)
        acc   = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        spec  = tn / (tn + fp + 1e-8)

        metrics[f"dice_c{c}"]  = dice
        metrics[f"iou_c{c}"]   = iou
        metrics[f"prec_c{c}"]  = prec
        metrics[f"rec_c{c}"]   = rec
        metrics[f"acc_c{c}"]   = acc
        metrics[f"spec_c{c}"]  = spec

    # macro averages (most important for papers)
    metrics["dice"]  = np.mean([metrics[f"dice_c{c}"] for c in range(1, num_classes)])  # foreground
    metrics["iou"]   = np.mean([metrics[f"iou_c{c}"]  for c in range(1, num_classes)])
    metrics["prec"]  = np.mean([metrics[f"prec_c{c}"] for c in range(1, num_classes)])
    metrics["rec"]   = np.mean([metrics[f"rec_c{c}"]  for c in range(1, num_classes)])
    metrics["acc"]   = metrics["acc_c0"]  # overall accuracy is fine
    return metrics


# ------------------------------------------------------------------
# 4. Model, loss, train / validate loops
# ------------------------------------------------------------------
def get_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,          # ← important: no pretrained weights
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model.to(DEVICE)


class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight
        self.bce = nn.BCEWithLogitsLoss() if NUM_CLASSES == 1 else nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        if NUM_CLASSES == 2:
            # multi-class
            ce = self.bce(logits, targets)
            probs = F.softmax(logits, dim=1)
            targets_onehot = F.one_hot(targets, NUM_CLASSES).permute(0,3,1,2).float()
            dims = (0,2,3)
            intersection = torch.sum(probs * targets_onehot, dims)
            cardinality  = torch.sum(probs + targets_onehot, dims)
            dice = (2. * intersection) / (cardinality + 1e-8)
            dice_loss = 1 - dice.mean()
            return self.weight * ce + (1 - self.weight) * dice_loss
        else:
            raise NotImplementedError


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_metrics = defaultdict(list)
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)
        preds = logits.argmax(1)
        batch_m = compute_metrics(preds, masks, NUM_CLASSES)
        for k, v in batch_m.items():
            all_metrics[k].append(v)
    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


# ------------------------------------------------------------------
# 5. Prepare folds (use original validation_fold if available)
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df = df[df["task"] == "he_tissue_segmentation"].copy()
if ORGAN:
    df = df[df["organ"] == ORGAN]

# keep only rows that actually exist in organised folders
valid_rows = []
for _, row in df.iterrows():
    subtype = str(row["histological_subtype"]).strip().lower().replace(" ", "_") or "unknown"
    stem = Path(row["name"]).stem
    img_p = Path(DATA_ROOT) / subtype / "images" / f"{stem}.png"
    if img_p.is_file():
        valid_rows.append(row)
df = pd.DataFrame(valid_rows).reset_index(drop=True)
print(f"Usable samples: {len(df)}")

# use the original 5 folds if present, otherwise create stratified folds
if "validation_fold" in df.columns and df["validation_fold"].notna().any():
    folds = []
    for f in sorted(df["validation_fold"].dropna().unique()):
        val_idx = df[df["validation_fold"] == f].index.tolist()
        train_idx = df[df["validation_fold"] != f].index.tolist()
        folds.append((train_idx, val_idx))
    print(f"Using original validation_fold → {len(folds)} folds")
else:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y = df["histological_subtype"].fillna("unknown")
    folds = list(skf.split(df, y))
    print("Created new stratified 5-fold CV")


# ------------------------------------------------------------------
# 6. 5-fold Cross-Validation
# ------------------------------------------------------------------
all_fold_results = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    print(f"\n{'='*60}\nFOLD {fold_idx+1}/5\n{'='*60}")

    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]

    train_ds = HESegDataset(train_df, DATA_ROOT, transform=train_tf, img_size=IMG_SIZE)
    val_ds   = HESegDataset(val_df,   DATA_ROOT, transform=val_tf,   img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = get_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = DiceBCELoss(weight=0.5)

    best_dice = 0.0
    history = []

    for epoch in range(1, NUM_EPOCHS+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_m   = evaluate(model, val_loader)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": tr_loss, **val_m})
        print(f"Epoch {epoch:03d} | loss={tr_loss:.4f} | "
              f"Dice={val_m['dice']:.4f} | IoU={val_m['iou']:.4f}")

        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            torch.save(model.state_dict(), OUTPUT / f"best_fold{fold_idx}.pth")

    # final metrics of the best model
    model.load_state_dict(torch.load(OUTPUT / f"best_fold{fold_idx}.pth"))
    final_m = evaluate(model, val_loader)
    final_m["fold"] = fold_idx
    all_fold_results.append(final_m)

    # save per-fold history
    pd.DataFrame(history).to_csv(OUTPUT / f"history_fold{fold_idx}.csv", index=False)

# ------------------------------------------------------------------
# 7. Statistical summary (exactly what reviewers expect)
# ------------------------------------------------------------------
results_df = pd.DataFrame(all_fold_results)
print("\n" + "="*60)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("="*60)

metrics_of_interest = ["dice", "iou", "prec", "rec", "acc"]
summary = {}
for m in metrics_of_interest:
    vals = results_df[m].values
    mean = vals.mean()
    std  = vals.std(ddof=1)
    ci   = stats.t.interval(0.95, len(vals)-1, loc=mean, scale=stats.sem(vals))
    summary[m] = {
        "mean": mean,
        "std": std,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "values": vals.tolist()
    }
    print(f"{m.upper():8s}: {mean:.4f} ± {std:.4f}   "
          f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

# save everything
results_df.to_csv(OUTPUT / "cv_results_per_fold.csv", index=False)
with open(OUTPUT / "cv_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAll results saved to → {OUTPUT.resolve()}")