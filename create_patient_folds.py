
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

root = Path(r"D:\microscope project")
source = root / "Data" / "data_overview.csv"
destination = root / "Data" / "data_overview_patient_folds.csv"

df = pd.read_csv(source)
df["validation_fold"] = df["validation_fold"].astype("object")

train_mask = (
    (df["task"] == "he_tissue_segmentation")
    & (df["split"] == "train")
)

train_df = df.loc[train_mask].copy()

if train_df["patient_id"].isna().any():
    raise ValueError("Some training images have no patient_id")

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for fold, (train_idx, val_idx) in enumerate(
    splitter.split(
        train_df,
        y=train_df["histological_subtype"].fillna("unknown"),
        groups=train_df["patient_id"]
    )
):
    val_rows = train_df.iloc[val_idx]
    df.loc[val_rows.index, "validation_fold"] = f"fold{fold}"

    train_patients = set(train_df.iloc[train_idx]["patient_id"])
    val_patients = set(val_rows["patient_id"])

    assert train_patients.isdisjoint(val_patients)

    print(
        f"Fold {fold}: "
        f"{len(train_idx)} training images, "
        f"{len(val_idx)} validation images, "
        f"{len(val_patients)} validation patients"
    )

assert df.loc[train_mask, "validation_fold"].notna().all()
assert df.loc[
    (df["task"] == "he_tissue_segmentation")
    & (df["split"] == "test"),
    "validation_fold"
].isna().all()

df.to_csv(destination, index=False)

print("\nSaved corrected manifest to:", destination)
print("Original manifest remains unchanged.")

