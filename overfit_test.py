from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import ignite_unet_pipeline as p

torch.manual_seed(42)

# Use only Fold 1 training images
train_idx, _ = p.folds[1]
train_df = p.df.iloc[train_idx].copy()

selected_names = [
    "patient65_he_roi1",
    "patient64_he_roi1",
    "patient64_he_roi2",
    "patient69_he_roi1",
]

small_df = train_df[
    train_df["name"].astype(str).map(
        lambda name: Path(name).stem in selected_names
    )
].copy()

assert len(small_df) == 4, (
    f"Expected 4 images, found {len(small_df)}"
)

# No random augmentation for memorization test
dataset = p.HESegDataset(
    small_df,
    p.DATA_ROOT,
    transform=p.val_tf,
    img_size=p.IMG_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

model = p.get_model("none")
criterion = p.DiceBCELoss(
    weight=0.5,
    background_weight=5.0
).to(p.DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0003,
    weight_decay=0.0
)

output_dir = Path(p.OUTPUT_DIR) / "overfit_test"
output_dir.mkdir(parents=True, exist_ok=True)

history = []

print("Selected images:")
print(small_df["name"].to_string(index=False))

for epoch in range(1, 61):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(p.DEVICE)
        masks = masks.to(p.DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()

    with torch.no_grad():
        images, masks = next(iter(loader))
        images = images.to(p.DEVICE)
        masks = masks.to(p.DEVICE)

        predictions = model(images).argmax(dim=1)

        metrics = p.compute_metrics(
            predictions,
            masks,
            p.NUM_CLASSES
        )

        predicted_bg = (
            predictions == 0
        ).float().mean().item() * 100

        actual_bg = (
            masks == 0
        ).float().mean().item() * 100

    record = {
        "epoch": epoch,
        "loss": total_loss,
        "background_dice": metrics["dice_c0"],
        "tissue_dice": metrics["dice_c1"],
        "macro_dice": metrics["macro_dice"],
        "actual_background_pct": actual_bg,
        "predicted_background_pct": predicted_bg
    }

    history.append(record)

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss={total_loss:.4f} | "
            f"BG Dice={metrics['dice_c0']:.4f} | "
            f"Tissue Dice={metrics['dice_c1']:.4f} | "
            f"Macro Dice={metrics['macro_dice']:.4f} | "
            f"Pred BG={predicted_bg:.2f}%"
        )

pd.DataFrame(history).to_csv(
    output_dir / "overfit_history.csv",
    index=False
)

torch.save(
    model.state_dict(),
    output_dir / "overfit_model.pth"
)

print("\nOverfitting test completed.")
print("Results:", output_dir)
