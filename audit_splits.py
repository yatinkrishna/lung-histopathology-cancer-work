import pandas as pd

path = r"D:\microscope project\Data\data_overview.csv"
df = pd.read_csv(path)
df = df[df["task"] == "he_tissue_segmentation"].copy()

print("\nSPLIT VS VALIDATION FOLD")
print(pd.crosstab(
    df["split"].fillna("MISSING"),
    df["validation_fold"].fillna("MISSING")
).to_string())

print("\nPATIENT OVERLAP BETWEEN TRAIN AND TEST")
train_patients = set(df.loc[df["split"] == "train", "patient_id"].dropna())
test_patients = set(df.loc[df["split"] == "test", "patient_id"].dropna())
overlap = train_patients & test_patients

print("Train patients:", len(train_patients))
print("Test patients:", len(test_patients))
print("Overlapping patients:", len(overlap))
print("Overlapping IDs:", sorted(map(str, overlap))[:20])

print("\nPATIENT OVERLAP ACROSS VALIDATION FOLDS")
assigned = df[
    (df["split"] == "train") &
    df["validation_fold"].notna()
]

for fold in sorted(assigned["validation_fold"].unique()):
    val = set(assigned.loc[assigned["validation_fold"] == fold, "patient_id"])
    tr = set(assigned.loc[assigned["validation_fold"] != fold, "patient_id"])
    print(fold, "overlapping patients:", len(val & tr))
