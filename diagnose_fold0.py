from pathlib import Path
import torch
import numpy as np
import pandas as pd
import cv2

import ignite_unet_pipeline as p

checkpoint = Path(
    r"D:\microscope project\Data\unet_results_camera_pilot\none\best_fold0.pth"
)

assert checkpoint.exists(), f"Checkpoint missing: {checkpoint}"

# Use the same patient-separated validation fold
df = p.df
train_idx, val_idx = p.folds[0]
val_df = df.iloc[val_idx]

dataset = p.HESegDataset(
    val_df,
    p.DATA_ROOT,
    transform=p.val_tf,
    img_size=p.IMG_SIZE
)

model = p.get_model("none")
model.load_state_dict(
    torch.load(checkpoint, map_location=p.DEVICE, weights_only=True)
)
model.eval()

output_dir = Path(
    r"D:\microscope project\Data\unet_results_camera_pilot\diagnostics"
)
output_dir.mkdir(parents=True, exist_ok=True)

records = []

with torch.no_grad():
    for i in range(len(dataset)):
        image, mask = dataset[i]

        logits = model(image.unsqueeze(0).to(p.DEVICE))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        gt = mask.cpu().numpy()

        gt_bg = gt == 0
        pred_bg = pred == 0

        intersection = np.logical_and(gt_bg, pred_bg).sum()
        denominator = gt_bg.sum() + pred_bg.sum()

        bg_dice = (
            2 * intersection / denominator
            if denominator > 0 else 1.0
        )

        records.append({
            "name": val_df.iloc[i]["name"],
            "gt_background_pct": 100 * gt_bg.mean(),
            "pred_background_pct": 100 * pred_bg.mean(),
            "background_dice": bg_dice,
        })

        if i < 10:
            img = image.permute(1, 2, 0).cpu().numpy()
            img = (img * p.STD + p.MEAN).clip(0, 1)
            img = (img * 255).astype(np.uint8)

            gt_vis = (gt * 255).astype(np.uint8)
            pred_vis = (pred * 255).astype(np.uint8)

            comparison = np.concatenate([
                img,
                cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2RGB),
                cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2RGB)
            ], axis=1)

            cv2.imwrite(
                str(output_dir / f"comparison_{i:02d}.png"),
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            )

results = pd.DataFrame(records)
results.to_csv(output_dir / "background_diagnostics.csv", index=False)

print("\n===== FOLD 0 BACKGROUND DIAGNOSTICS =====")
print("Validation images:", len(results))
print("Mean GT background %:", round(results["gt_background_pct"].mean(), 4))
print("Mean predicted background %:", round(results["pred_background_pct"].mean(), 4))
print("Mean background Dice:", round(results["background_dice"].mean(), 6))
print("Images with zero predicted background:",
      int((results["pred_background_pct"] == 0).sum()))

print("\nWorst 10 images:")
print(results.sort_values("background_dice").head(10).to_string(index=False))

print("\nSaved diagnostics to:", output_dir)
