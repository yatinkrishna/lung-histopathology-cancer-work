# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:42:44 2026

@author: Pooja

============================================================================
IGNITE H&E tissue segmentation: attention ablation + prediction, all-in-one.
============================================================================
This is the merged version of three previously separate files:
  - unet_models.py               (model / attention-mechanism definitions)
  - unet_attention_ablation.py   (5-fold CV training across ablation arms)
  - predict_segmentation.py      (out-of-fold predicted mask export)

Set RUN_MODE below to control what running this file does:
  "train"   -> train every arm in MODEL_TYPES, 5-fold CV each (as before)
  "predict" -> export a predicted mask for every image, out-of-fold, using
               already-trained checkpoints for PREDICT_MODEL_TYPE
  "both"    -> train first, then immediately run prediction

Because everything now lives in one file, df/folds are built exactly once
and reused by both training and prediction - so there is no risk of the
two ever using a different ORGAN/SEED/fold split by accident (previously
a manual "must match" requirement between the separate scripts).
"""

# ============================================================
# CONFIG - edit these lines
# ============================================================
RUN_MODE = "train"   # "train" | "predict" | "both"

DATA_ROOT       = r"E:\ignite tool kit he segmentation\Data\organised_he"
CSV_PATH        = r"E:\ignite tool kit he segmentation\Data\data_overview.csv"
OUTPUT_DIR      = r"E:\ignite tool kit he segmentation\Data\unet_results"
PRED_OUTPUT_DIR = r"E:\ignite tool kit he segmentation\Data\unet_results\predictions"

ORGAN          = "lung"
NUM_CLASSES    = 2
BATCH_SIZE     = 16          # good starting point for 16-24 GB VRAM
NUM_EPOCHS     = 50
LR             = 1e-4
IMG_SIZE       = 512
NUM_WORKERS    = 0           # safe on Windows with recent PyTorch
SEED           = 42
DEVICE         = "cuda"

# --- Ablation arms (used when RUN_MODE is "train" or "both") -------
# "none"            -> plain ResNet34-UNet, no attention (baseline)
# "attention_gate"  -> Attention Gates at skip connections (Oktay et al. 2018)
# "se"              -> Squeeze-and-Excitation channel attention on each skip
#                       (Hu et al. 2018)
# "cbam"            -> Convolutional Block Attention Module: channel + spatial
#                       attention on each skip (Woo et al. 2018)
# "eca"             -> Efficient Channel Attention on each skip, no dimension
#                       reduction (Wang et al. 2020)
# "self_attention"  -> Non-local / self-attention block, applied ONCE at the
#                       bottleneck only (Wang et al. 2018) - full-resolution
#                       self-attention at the shallow skip levels is O(HW^2)
#                       and not tractable at 512x512, so this is the standard
#                       placement used in practice.
#
# All arms share the identical encoder/decoder/data/loss/folds - only the
# attention module (or its absence) differs, so any performance delta is
# attributable to the attention mechanism itself.
MODEL_TYPES = ["none", "attention_gate", "se", "cbam", "eca", "self_attention"]

# --- Prediction settings (used when RUN_MODE is "predict" or "both") ----
PREDICT_MODEL_TYPE = "attention_gate"   # which arm's checkpoints to predict with
SAVE_OVERLAYS = True
OVERLAY_ALPHA = 0.45
# ============================================================
# ============================================================

import os, json, random, warnings, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as tv_models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

warnings.filterwarnings("ignore")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

OUTPUT = Path(OUTPUT_DIR)
OUTPUT.mkdir(parents=True, exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
# 3. Metrics (vectorised, publication-ready; batch-level, used in training)
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


def compute_single_image_metrics(pred, gt):
    """Foreground dice/iou for ONE image (numpy arrays, NUM_CLASSES==2 case).
    Used by prediction/export, as opposed to compute_metrics() above which
    is batch-level and used during training."""
    p = pred.astype(bool)
    t = gt.astype(bool)
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return float(dice), float(iou)


# ------------------------------------------------------------------
# 4. Model definitions - ResNet34-UNet backbone + 5 interchangeable
#    attention mechanisms, all sharing the same encoder/decoder skeleton.
# ------------------------------------------------------------------
class ResNet34Encoder(nn.Module):
    """Same backbone smp.Unet(encoder_name='resnet34') used, exposed as
    intermediate feature maps so a custom decoder (with any attention
    mechanism, or none) can be attached identically across ablation arms."""
    def __init__(self, pretrained=False):
        super().__init__()
        base = tv_models.resnet34(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu)  # 64ch,  /2
        self.pool   = base.maxpool                                     #        /4
        self.layer1 = base.layer1   # 64ch,  /4
        self.layer2 = base.layer2   # 128ch, /8
        self.layer3 = base.layer3   # 256ch, /16
        self.layer4 = base.layer4   # 512ch, /32

    def forward(self, x):
        x1 = self.stem(x)                # 64ch,  /2
        x2 = self.layer1(self.pool(x1))  # 64ch,  /4
        x3 = self.layer2(x2)             # 128ch, /8
        x4 = self.layer3(x3)             # 256ch, /16
        x5 = self.layer4(x4)             # 512ch, /32
        return [x1, x2, x3, x4, x5]      # ascending resolution -> bottleneck


class AttentionGate(nn.Module):
    """Additive attention gate, Oktay et al. 2018 ('Attention U-Net').
    Gating signal g (coarser decoder features) suppresses irrelevant
    regions of the skip connection x before it's concatenated."""
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)          # (B,1,H,W) attention coefficients in [0,1]
        return x * psi


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al. 2018)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    """Channel-attention half of CBAM."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial-attention half of CBAM."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al. 2018):
    channel attention then spatial attention, applied sequentially."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class ECABlock(nn.Module):
    """Efficient Channel Attention (Wang et al. 2020): channel attention
    via a single 1D conv across channels, no dimensionality reduction -
    much cheaper than SE, and shown to be at least as effective."""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class NonLocalBlock2D(nn.Module):
    """Non-local / self-attention block (Wang et al. 2018), embedded
    Gaussian variant. Global pairwise attention over all spatial
    positions - applied once at the bottleneck only, since cost is
    O((HW)^2) and the bottleneck is the only feature map small enough
    (16x16 = 256 tokens at IMG_SIZE=512) to be practical."""
    def __init__(self, channels, reduction=2):
        super().__init__()
        inter_channels = max(channels // reduction, 1)
        self.theta = nn.Conv2d(channels, inter_channels, 1)
        self.phi   = nn.Conv2d(channels, inter_channels, 1)
        self.g     = nn.Conv2d(channels, inter_channels, 1)
        self.out   = nn.Conv2d(inter_channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # learnable residual scale, starts at 0

    def forward(self, x):
        b, c, h, w = x.shape
        theta = self.theta(x).view(b, -1, h * w).permute(0, 2, 1)   # B,HW,inter
        phi   = self.phi(x).view(b, -1, h * w)                       # B,inter,HW
        g     = self.g(x).view(b, -1, h * w).permute(0, 2, 1)        # B,HW,inter
        attn  = torch.softmax(torch.bmm(theta, phi), dim=-1)         # B,HW,HW
        out   = torch.bmm(attn, g)                                    # B,HW,inter
        out   = out.permute(0, 2, 1).contiguous().view(b, -1, h, w)
        out   = self.out(out)
        return x + self.gamma * out


# Skip-level attention types (applied per skip connection). "attention_gate"
# is handled separately inside DecoderBlock since it needs the gating signal.
SKIP_ATTENTION_FACTORY = {
    "se":   SEBlock,
    "cbam": CBAM,
    "eca":  ECABlock,
}


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, attention_type="none"):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.attention_type = attention_type
        self.skip_attn = None
        self.gate = None

        if skip_channels > 0:
            if attention_type == "attention_gate":
                self.gate = AttentionGate(f_g=in_channels, f_l=skip_channels,
                                           f_int=max(skip_channels // 2, 1))
            elif attention_type in SKIP_ATTENTION_FACTORY:
                self.skip_attn = SKIP_ATTENTION_FACTORY[attention_type](skip_channels)
            # "none" and "self_attention" (handled at the bottleneck, not per-skip)
            # leave the skip connection unmodified here.

        conv_in = in_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            if self.gate is not None:
                skip = self.gate(g=x, x=skip)
            elif self.skip_attn is not None:
                skip = self.skip_attn(skip)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels=(64, 64, 128, 256, 512),
                 decoder_channels=(256, 128, 64, 32, 16), attention_type="none"):
        super().__init__()
        enc_ch = list(encoder_channels[::-1])          # [512,256,128,64,64] bottleneck->shallow
        skip_chs = enc_ch[1:] + [0]                     # last block has no skip

        self.bottleneck_attn = None
        if attention_type == "self_attention":
            self.bottleneck_attn = NonLocalBlock2D(enc_ch[0])

        blocks, in_ch = [], enc_ch[0]
        for out_ch, skip_ch in zip(decoder_channels, skip_chs):
            blocks.append(DecoderBlock(in_ch, skip_ch, out_ch, attention_type=attention_type))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):
        feats = features[::-1]           # [x5(bottleneck), x4, x3, x2, x1]
        x = feats[0]
        if self.bottleneck_attn is not None:
            x = self.bottleneck_attn(x)
        skips = feats[1:] + [None]
        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)
        return x


class AttentionUNet(nn.Module):
    """ResNet34-UNet. attention_type selects which mechanism (if any) is
    used; see MODEL_TYPES comment above for the full list. Everything
    else (encoder, decoder channel widths, conv blocks) is identical
    across arms."""
    def __init__(self, num_classes=2, attention_type="none", pretrained=False):
        super().__init__()
        self.encoder = ResNet34Encoder(pretrained=pretrained)
        self.decoder = UNetDecoder(attention_type=attention_type)
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)
        return self.head(out)


def build_model(num_classes, attention_type="none", pretrained=False, device="cuda"):
    model = AttentionUNet(num_classes=num_classes, attention_type=attention_type,
                           pretrained=pretrained)
    return model.to(device)


def get_model(model_type):
    return build_model(num_classes=NUM_CLASSES, attention_type=model_type,
                        pretrained=False, device=DEVICE)


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
# 5. Prepare df + folds (use original validation_fold if available).
#    Built ONCE here and reused by both training and prediction below.
# ------------------------------------------------------------------
def build_df_and_folds():
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
        print(f"Using original validation_fold -> {len(folds)} folds")
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        y = df["histological_subtype"].fillna("unknown")
        folds = list(skf.split(df, y))
        print("Created new stratified 5-fold CV")

    return df, folds


df, folds = build_df_and_folds()


# ------------------------------------------------------------------
# 6. 5-fold Cross-Validation, run once per ablation arm (model_type)
# ------------------------------------------------------------------
metrics_of_interest = ["dice", "iou", "prec", "rec", "acc"]


def run_cv_for_model(model_type):
    model_out = OUTPUT / model_type
    model_out.mkdir(parents=True, exist_ok=True)
    all_fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*60}\n[{model_type}] FOLD {fold_idx+1}/{len(folds)}\n{'='*60}")

        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        train_ds = HESegDataset(train_df, DATA_ROOT, transform=train_tf, img_size=IMG_SIZE)
        val_ds   = HESegDataset(val_df,   DATA_ROOT, transform=val_tf,   img_size=IMG_SIZE)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)

        model = get_model(model_type)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        criterion = DiceBCELoss(weight=0.5)

        best_dice = -1.0  # not 0.0: guarantees a checkpoint is saved even if
                          # the first epoch's Dice is exactly 0.0 (degenerate fold)
        history = []
        ckpt_path = model_out / f"best_fold{fold_idx}.pth"

        for epoch in range(1, NUM_EPOCHS+1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_m   = evaluate(model, val_loader)
            scheduler.step()

            history.append({"epoch": epoch, "train_loss": tr_loss, **val_m})
            print(f"Epoch {epoch:03d} | loss={tr_loss:.4f} | "
                  f"Dice={val_m['dice']:.4f} | IoU={val_m['iou']:.4f}")

            if val_m["dice"] > best_dice:
                best_dice = val_m["dice"]
                torch.save(model.state_dict(), ckpt_path)

        # final metrics of the best model
        model.load_state_dict(torch.load(ckpt_path))
        final_m = evaluate(model, val_loader)
        final_m["fold"] = fold_idx
        final_m["model_type"] = model_type
        all_fold_results.append(final_m)

        # save per-fold history
        pd.DataFrame(history).to_csv(model_out / f"history_fold{fold_idx}.csv", index=False)

    results_df = pd.DataFrame(all_fold_results)
    print(f"\n{'='*60}\n{model_type.upper()} - {len(folds)}-FOLD CV RESULTS\n{'='*60}")

    summary = {}
    for m in metrics_of_interest:
        vals = results_df[m].values
        mean, std = vals.mean(), vals.std(ddof=1)
        ci = stats.t.interval(0.95, len(vals)-1, loc=mean, scale=stats.sem(vals))
        summary[m] = {"mean": mean, "std": std, "ci_low": ci[0], "ci_high": ci[1],
                      "values": vals.tolist()}
        print(f"{m.upper():8s}: {mean:.4f} +/- {std:.4f}   95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

    results_df.to_csv(model_out / "cv_results_per_fold.csv", index=False)
    with open(model_out / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results_df, summary


def run_ablation():
    """Runs every arm in MODEL_TYPES, then compares each against the
    'none' baseline (if present) with a paired t-test across folds."""
    all_results = {}
    all_summaries = {}
    for model_type in MODEL_TYPES:
        res_df, summ = run_cv_for_model(model_type)
        all_results[model_type] = res_df
        all_summaries[model_type] = summ

    combined_df = pd.concat(all_results.values(), ignore_index=True)
    combined_df.to_csv(OUTPUT / "ablation_all_results.csv", index=False)

    print("\n" + "="*60)
    print("ABLATION SUMMARY: attention vs. no-attention")
    print("="*60)
    for model_type, summ in all_summaries.items():
        line = "  ".join(f"{m.upper()}={summ[m]['mean']:.4f}+/-{summ[m]['std']:.4f}"
                          for m in metrics_of_interest)
        print(f"{model_type:15s}: {line}")

    # Paired comparison against the "none" baseline (same folds seen by every
    # arm) - only meaningful if "none" is one of the arms that were run.
    comparison = {}
    if "none" in all_results:
        base_df = all_results["none"].sort_values("fold").reset_index(drop=True)
        for model_type, res_df in all_results.items():
            if model_type == "none":
                continue
            cand_df = res_df.sort_values("fold").reset_index(drop=True)
            if len(cand_df) != len(base_df):
                continue
            print(f"\nPaired comparison ({model_type} vs none, {len(base_df)} matched folds):")
            comparison[model_type] = {}
            for m in metrics_of_interest:
                t_stat, p_val = stats.ttest_rel(cand_df[m].values, base_df[m].values)
                delta = cand_df[m].values.mean() - base_df[m].values.mean()
                comparison[model_type][m] = {"delta": float(delta), "t_stat": float(t_stat), "p_value": float(p_val)}
                sig = "*" if p_val < 0.05 else ""
                print(f"  {m.upper():8s}: delta={delta:+.4f}  t={t_stat:.3f}  p={p_val:.4f} {sig}")
        with open(OUTPUT / "ablation_paired_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
    else:
        print("\n(No 'none' baseline arm was run, so no paired comparison was computed. "
              "Add 'none' to MODEL_TYPES to get baseline-relative deltas and p-values.)")

    print(f"\nAll results saved to -> {OUTPUT.resolve()}")


# ------------------------------------------------------------------
# 7. Out-of-fold prediction / mask export for every image.
#    Each image is predicted by the checkpoint from the ONE fold where
#    it was held out - never by a model that trained on it.
# ------------------------------------------------------------------
def preprocess_image(image_rgb):
    """Resize + ImageNet-normalize, matching val_tf above (minus
    augmentation, since this is inference)."""
    resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    normed = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def draw_overlay(image_rgb, mask, alpha=OVERLAY_ALPHA):
    overlay = image_rgb.copy()
    overlay[mask > 0] = (0.3 * overlay[mask > 0] + 0.7 * np.array([0, 255, 0])).astype(np.uint8)
    blended = (alpha * overlay + (1 - alpha) * image_rgb).astype(np.uint8)
    return Image.fromarray(blended)


def run_prediction(model_type):
    model_ckpt_dir = OUTPUT / model_type
    out_dir = Path(PRED_OUTPUT_DIR) / model_type
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_OVERLAYS:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for fold_idx, (_, val_idx) in enumerate(folds):
        ckpt_path = model_ckpt_dir / f"best_fold{fold_idx}.pth"
        if not ckpt_path.exists():
            print(f"  [skip fold {fold_idx}] checkpoint not found: {ckpt_path}")
            continue

        model = build_model(num_classes=NUM_CLASSES, attention_type=model_type,
                             pretrained=False, device=DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()

        val_df = df.iloc[val_idx]
        print(f"\nFold {fold_idx}: predicting {len(val_df)} held-out image(s) "
              f"with {ckpt_path.name}")

        for _, row in val_df.iterrows():
            subtype = str(row["histological_subtype"]).strip().lower().replace(" ", "_") or "unknown"
            stem = Path(row["name"]).stem
            img_path = Path(DATA_ROOT) / subtype / "images" / f"{stem}.png"
            gt_path = Path(DATA_ROOT) / subtype / "masks" / f"{stem}.png"

            image = np.array(Image.open(img_path).convert("RGB"))
            orig_h, orig_w = image.shape[:2]

            tensor = preprocess_image(image).to(DEVICE)
            with torch.no_grad():
                logits = model(tensor)
                pred_512 = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

            # resize predicted mask back to the image's original resolution
            pred_full = cv2.resize(pred_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            mask_out_path = masks_dir / f"{stem}.png"
            Image.fromarray(pred_full).save(mask_out_path)

            row_result = {
                "name": row["name"],
                "fold": fold_idx,
                "model_type": model_type,
                "checkpoint": str(ckpt_path),
                "mask_path": str(mask_out_path),
            }

            if gt_path.exists():
                gt_mask = np.array(Image.open(gt_path))
                if NUM_CLASSES == 2:
                    gt_mask = (gt_mask > 0).astype(np.uint8)
                dice, iou = compute_single_image_metrics(pred_full, gt_mask)
                row_result["dice"] = dice
                row_result["iou"] = iou

            if SAVE_OVERLAYS:
                overlay_img = draw_overlay(image, pred_full)
                overlay_path = overlays_dir / f"{stem}.png"
                overlay_img.save(overlay_path)
                row_result["overlay_path"] = str(overlay_path)

            manifest_rows.append(row_result)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "predictions_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"\nSaved {len(manifest_df)} predicted mask(s) to: {masks_dir}")
    if SAVE_OVERLAYS:
        print(f"Saved overlays to: {overlays_dir}")
    print(f"Manifest: {manifest_path}")

    if "dice" in manifest_df.columns and len(manifest_df):
        print(f"\nOut-of-fold Dice: {manifest_df['dice'].mean():.4f} +/- {manifest_df['dice'].std():.4f}")
        print(f"Out-of-fold IoU:  {manifest_df['iou'].mean():.4f} +/- {manifest_df['iou'].std():.4f}")

        worst = manifest_df.sort_values("dice").head(10)
        print("\n10 lowest-Dice images (worth a manual look):")
        print(worst[["name", "fold", "dice", "iou"]].to_string(index=False))

    return manifest_df


# ------------------------------------------------------------------
# 8. Dispatch based on RUN_MODE
# ------------------------------------------------------------------
if __name__ == "__main__":
    if RUN_MODE in ("train", "both"):
        run_ablation()
    if RUN_MODE in ("predict", "both"):
        run_prediction(PREDICT_MODEL_TYPE)