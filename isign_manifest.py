from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(r"D:\ISLR_DATA\iSign")
CSV_PATH = BASE_DIR / "iSign_v1.1.csv"
POSE_ROOT = BASE_DIR / "iSign-poses_v1.1" / "iSign-poses_v1.1"
OUTPUT_DIR = BASE_DIR / "metadata"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}. Edit CSV_PATH at top.")
    if not POSE_ROOT.is_dir():
        raise FileNotFoundError(f"Pose directory not found: {POSE_ROOT}. Edit POSE_ROOT at top.")

    print(f"Reading annotations: {CSV_PATH}")
    annotations = pd.read_csv(CSV_PATH, dtype={"uid": "string", "text": "string"})
    if not {"uid", "text"}.issubset(annotations.columns):
        raise ValueError(f"Expected columns uid and text; got {list(annotations.columns)}")
    annotations["uid"] = annotations["uid"].str.strip()
    annotations["text"] = annotations["text"].str.strip()
    annotations.loc[annotations["text"].eq(""), "text"] = pd.NA
    if annotations["uid"].isna().any():
        raise ValueError("CSV has missing UIDs.")

    duplicates = annotations[annotations["uid"].duplicated(keep=False)]
    if not duplicates.empty:
        duplicates.to_csv(OUTPUT_DIR / "duplicate_csv_uids.csv", index=False)
        raise ValueError(f"Duplicate CSV UIDs found. See {OUTPUT_DIR / 'duplicate_csv_uids.csv'}")

    print(f"CSV rows: {len(annotations):,}")
    print(f"Scanning pose files: {POSE_ROOT}")
    pose_paths = sorted(POSE_ROOT.rglob("*.pose"))
    if not pose_paths:
        raise FileNotFoundError(f"No .pose files found under {POSE_ROOT}")

    poses = pd.DataFrame({
        "uid": [p.stem.strip() for p in pose_paths],
        "pose_path": [str(p.resolve()) for p in pose_paths],
    })
    print(f"Pose files found: {len(poses):,}")

    duplicate_poses = poses[poses["uid"].duplicated(keep=False)]
    if not duplicate_poses.empty:
        duplicate_poses.to_csv(OUTPUT_DIR / "duplicate_pose_uids.csv", index=False)
        raise ValueError(f"Duplicate pose filename stems found. See {OUTPUT_DIR / 'duplicate_pose_uids.csv'}")

    # UID format is video_id + final hyphen + segment number.
    # rsplit is deliberate: a UID such as _1tafiQHMEs--3 has video_id _1tafiQHMEs-
    # and segment number 3.
    uid_parts = poses["uid"].str.rsplit("-", n=1, expand=True)
    if uid_parts.shape[1] < 2:
        malformed = poses.copy()
        malformed.to_csv(OUTPUT_DIR / "malformed_uids.csv", index=False)
        raise ValueError("Could not parse segment numbers from pose UIDs.")

    poses["video_id"] = uid_parts[0]
    segment_numbers = pd.to_numeric(uid_parts[1], errors="coerce")
    malformed_mask = segment_numbers.isna()
    malformed = poses[malformed_mask].copy()
    if not malformed.empty:
        malformed.to_csv(OUTPUT_DIR / "malformed_uids.csv", index=False)
    poses["segment_index"] = segment_numbers
    poses = poses[~malformed_mask].copy()
    poses["segment_index"] = poses["segment_index"].astype("int64")

    merged = poses.merge(
        annotations, on="uid", how="left", indicator=True, validate="one_to_one"
    )
    merged["text_missing"] = merged["text"].isna() | merged["text"].str.strip().eq("").fillna(True)

    unmatched_pose = merged[merged["_merge"] == "left_only"].copy()
    matched_missing_text = merged[(merged["_merge"] == "both") & merged["text_missing"]].copy()
    csv_rows_without_pose = annotations[~annotations["uid"].isin(set(poses["uid"]))].copy()

    merged.to_csv(OUTPUT_DIR / "all_pose_matches.csv", index=False)
    unmatched_pose.to_csv(OUTPUT_DIR / "unmatched_pose_files.csv", index=False)
    matched_missing_text.to_csv(OUTPUT_DIR / "matched_poses_missing_text.csv", index=False)
    csv_rows_without_pose.to_csv(OUTPUT_DIR / "csv_rows_without_pose.csv", index=False)

    valid = merged[(merged["_merge"] == "both") & (~merged["text_missing"])][
        ["uid", "video_id", "segment_index", "pose_path", "text"]
    ].copy().reset_index(drop=True)

    if valid.empty:
        raise RuntimeError("No usable pose/text matches found.")
    if valid["video_id"].nunique() < 3:
        raise RuntimeError("Need at least 3 source videos to create train/validation/test splits.")

    # Group by source video to prevent segments from the same video leaking across splits.
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, temp_idx = next(splitter1.split(valid, groups=valid["video_id"]))
    train = valid.iloc[train_idx].copy()
    temp = valid.iloc[temp_idx].copy().reset_index(drop=True)

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=43)
    val_idx, test_idx = next(splitter2.split(temp, groups=temp["video_id"]))
    val = temp.iloc[val_idx].copy()
    test = temp.iloc[test_idx].copy()

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"
    manifest = pd.concat([train, val, test], ignore_index=True)
    manifest = manifest.sort_values(["split", "video_id", "segment_index", "uid"]).reset_index(drop=True)

    groups = {name: set(group["video_id"]) for name, group in manifest.groupby("split")}
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise RuntimeError("Video leakage detected between splits.")

    manifest_path = OUTPUT_DIR / "isign_translation_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("\nManifest summary")
    print("----------------")
    print(f"CSV rows:                       {len(annotations):,}")
    print(f"Pose files found:               {len(pose_paths):,}")
    print(f"Malformed pose UIDs:            {len(malformed):,}")
    print(f"Pose files without CSV UID:     {len(unmatched_pose):,}")
    print(f"CSV rows without pose:          {len(csv_rows_without_pose):,}")
    print(f"Matched poses missing text:     {len(matched_missing_text):,}")
    print(f"Usable pose-text pairs:         {len(manifest):,}")
    print(f"Distinct source videos:         {manifest['video_id'].nunique():,}")
    print("\nSplit counts")
    for name in ["train", "validation", "test"]:
        part = manifest[manifest["split"] == name]
        print(f"{name:10s} segments={len(part):,} videos={part['video_id'].nunique():,}")
    print(f"\nSaved manifest: {manifest_path}")
    print(f"Saved diagnostics in: {OUTPUT_DIR}")
    print("This is a group-based approximately 80/10/10 source-video split; segment proportions may differ.")
    print("It is not an official predefined iSign split.")

if __name__ == "__main__":
    main()
