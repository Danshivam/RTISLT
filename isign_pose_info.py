"""
Inspect one iSign .pose file and its matching translation CSV row.

Run from the ISLR project root:
    python inspect_isign_pose.py "D:\ISLR_DATA\iSign\poses\zyvXu0nLgFI--18.pose"

Optional CSV path:
    python inspect_isign_pose.py "D:\path\sample.pose" --csv "D:\ISLR_DATA\iSign\iSign_v1.1.csv"
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from pose_format import Pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pose_file", help="Full path to one .pose file")
    parser.add_argument(
        "--csv",
        default=r"D:\ISLR_DATA\iSign\iSign_v1.1.csv",
        help="Path to iSign translation CSV",
    )
    args = parser.parse_args()

    pose_path = Path(args.pose_file)
    csv_path = Path(args.csv)

    if not pose_path.is_file():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # A UID is the whole filename without the .pose extension.
    uid = pose_path.stem

    annotations = pd.read_csv(csv_path)
    if not {"uid", "text"}.issubset(annotations.columns):
        raise ValueError(
            f"Expected CSV columns 'uid' and 'text'; got {list(annotations.columns)}"
        )

    annotations["uid"] = annotations["uid"].astype(str).str.strip()
    matched = annotations.loc[annotations["uid"] == uid, "text"]

    print("=" * 68)
    print("iSign pose inspection")
    print("=" * 68)
    print(f"Pose file : {pose_path}")
    print(f"UID       : {uid}")
    print(f"CSV rows matching UID: {len(matched)}")
    if len(matched):
        text = matched.iloc[0]
        print(f"English text: {text if pd.notna(text) else '[MISSING]'}")
    else:
        print("English text: [NO CSV UID MATCH]")

    # Official iSign files use the pose-format binary format.
    pose = Pose.read(pose_path.read_bytes())

    data = np.asarray(pose.body.data)
    confidence = np.asarray(pose.body.confidence)

    print("\nPose arrays")
    print("-" * 68)
    print(f"pose.body.data       shape={data.shape}, dtype={data.dtype}")
    print(
        f"pose.body.confidence shape={confidence.shape}, "
        f"dtype={confidence.dtype}"
    )
    print(f"Data finite values   : {np.isfinite(data).sum():,}/{data.size:,}")
    print(f"Data NaN values      : {np.isnan(data).sum():,}")
    print(f"Data Inf values      : {np.isinf(data).sum():,}")

    finite_data = data[np.isfinite(data)]
    if finite_data.size:
        print(f"Data min / max       : {finite_data.min():.6f} / {finite_data.max():.6f}")

    finite_conf = confidence[np.isfinite(confidence)]
    if finite_conf.size:
        print(
            "Confidence min / max : "
            f"{finite_conf.min():.6f} / {finite_conf.max():.6f}"
        )

    # Print header representation without assuming every header version
    # exposes exactly the same attributes.
    print("\nHeader metadata")
    print("-" * 68)
    header = getattr(pose, "header", None)
    if header is None:
        print("No header attribute exposed by this installed pose-format version.")
    else:
        print(header)

    print("\nNext: use the printed shape to define the translation feature adapter.")
    print("Do not assume the pose feature dimension is 225; inspect the real format first.")


if __name__ == "__main__":
    main()
