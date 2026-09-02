from datasets import load_dataset
import pandas as pd
import os

OUTPUT_DIR = "data/include/metadata"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "include50_manifest.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("Loading INCLUDE metadata...")

ds = load_dataset(
    "ai4bharat/INCLUDE"
)

# --------------------------------------------------
# Combine official splits
# --------------------------------------------------

frames = []

for split in ["train", "val", "test"]:

    df = ds[split].to_pandas()

    df["split"] = split

    frames.append(df)

df = pd.concat(
    frames,
    ignore_index=True
)

# --------------------------------------------------
# Keep INCLUDE-50
# --------------------------------------------------

df = df[
    df["include_50"] == True
].copy()

# --------------------------------------------------
# Useful columns
# --------------------------------------------------

df = df[
    [
        "parent_label",
        "label",
        "video_path",
        "include_50",
        "split"
    ]
]


# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("======================================")
print("INCLUDE-50 MANIFEST")
print("======================================")

print(
    "Total videos:",
    len(df)
)

print(
    "Train:",
    sum(df["split"] == "train")
)

print(
    "Validation:",
    sum(df["split"] == "val")
)

print(
    "Test:",
    sum(df["split"] == "test")
)

print(
    "Classes:",
    df["label"].nunique()
)

print()
print("Saved:")
print(OUTPUT_FILE)