import os
import time
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST_PATH = "data/include/metadata/include50_manifest.csv"

VIDEO_ROOT = r"D:\ISLR_DATA\INCLUDE\videos"

OUTPUT_ROOT = r"data\landmarks\include50"

MODEL_PATH = r"models\holistic_landmarker.task"

# We are intentionally processing only 10 videos for testing.
MAX_VIDEOS = None


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HolisticLandmarker = (
    mp.tasks.vision.HolisticLandmarker
)

HolisticLandmarkerOptions = (
    mp.tasks.vision.HolisticLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(landmarks, expected_count):
    """
    Convert a MediaPipe landmark collection into
    a flat [x, y, z] vector.

    If the landmark group is missing, return zeros.
    """

    if not landmarks:
        return np.zeros(
            expected_count * 3,
            dtype=np.float32
        )

    values = []

    for landmark in landmarks:
        values.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(
        values,
        dtype=np.float32
    )


# ============================================================
# EXTRACT FEATURES FROM ONE FRAME
# ============================================================

def extract_frame_features(result):
    """
    Convert one MediaPipe result into
    a 225-dimensional feature vector.

    63  -> left hand
    63  -> right hand
    99  -> body pose
    ----------------
    225 -> total
    """

    left_hand = extract_landmarks(
        result.left_hand_landmarks,
        21
    )

    right_hand = extract_landmarks(
        result.right_hand_landmarks,
        21
    )

    pose = extract_landmarks(
        result.pose_landmarks,
        33
    )

    features = np.concatenate([
        left_hand,
        right_hand,
        pose
    ])

    return features.astype(np.float32)


# ============================================================
# NORMALIZATION
# ============================================================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

POSE_START = 126


def normalize_frame(frame):
    """
    Center the signer around the midpoint of the shoulders
    and normalize scale using shoulder distance.

    Missing landmarks remain [0, 0, 0].
    """

    frame = frame.copy()

    # --------------------------------------------
    # Locate shoulder coordinates
    # --------------------------------------------

    left_index = (
        POSE_START
        + LEFT_SHOULDER * 3
    )

    right_index = (
        POSE_START
        + RIGHT_SHOULDER * 3
    )

    left_shoulder = frame[
        left_index:left_index + 3
    ]

    right_shoulder = frame[
        right_index:right_index + 3
    ]

    # --------------------------------------------
    # Cannot normalize without shoulders
    # --------------------------------------------

    if (
        np.allclose(left_shoulder, 0)
        or
        np.allclose(right_shoulder, 0)
    ):
        return frame

    # --------------------------------------------
    # Shoulder midpoint
    # --------------------------------------------

    center = (
        left_shoulder
        + right_shoulder
    ) / 2.0

    # --------------------------------------------
    # Shoulder distance
    # --------------------------------------------

    scale = np.linalg.norm(
        left_shoulder
        - right_shoulder
    )

    if scale < 1e-6:
        return frame

    # --------------------------------------------
    # Landmark groups
    # --------------------------------------------

    groups = [
        (0, 63),      # left hand
        (63, 126),    # right hand
        (126, 225)    # pose
    ]

    # --------------------------------------------
    # Normalize each landmark
    # --------------------------------------------

    for start, end in groups:

        for i in range(
            start,
            end,
            3
        ):

            point = frame[
                i:i + 3
            ]

            # Preserve missing landmarks
            if np.allclose(point, 0):
                continue

            frame[
                i:i + 3
            ] = (
                point - center
            ) / scale

    return frame


# ============================================================
# NORMALIZE ENTIRE VIDEO
# ============================================================

def normalize_sequence(sequence):
    """
    Normalize every frame in the sequence.
    """

    normalized = np.zeros_like(
        sequence,
        dtype=np.float32
    )

    for i in range(
        len(sequence)
    ):
        normalized[i] = normalize_frame(
            sequence[i]
        )

    return normalized


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(
    video_path,
    output_path,
    landmarker,
    start_timestamp_ms
):
    """
    Read one video frame-by-frame,
    extract landmarks, normalize them,
    and save the result as .npy.
    """

    print()
    print("-" * 60)
    print("Processing:")
    print(video_path)

    # --------------------------------------------
    # Open video
    # --------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video."
        )

    sequence = []

    # --------------------------------------------
    # Timestamp
    # --------------------------------------------

    frame_count = 0

    # --------------------------------------------
    # Process frames
    # --------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        # ----------------------------------------
        # OpenCV BGR → RGB
        # ----------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------
        # MediaPipe image
        # ----------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ----------------------------------------
        # Increasing timestamp
        # ----------------------------------------

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25.0

        frame_interval_ms = 1000.0 / fps


        timestamp_ms = int(
            start_timestamp_ms
            + frame_count * frame_interval_ms
        )

        # ----------------------------------------
        # Run MediaPipe
        # ----------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ----------------------------------------
        # Convert to 225 features
        # ----------------------------------------

        features = extract_frame_features(
            result
        )

        sequence.append(
            features
        )

        frame_count += 1

    cap.release()

    # --------------------------------------------
    # Convert to NumPy
    # --------------------------------------------

    if not sequence:
        raise RuntimeError(
            "No frames found in video."
        )

    sequence = np.array(
        sequence,
        dtype=np.float32
    )

    print(
        "Raw shape:",
        sequence.shape
    )

    # --------------------------------------------
    # Normalize
    # --------------------------------------------

    normalized = normalize_sequence(
        sequence
    )

    print(
        "Normalized shape:",
        normalized.shape
    )

    print(
        "NaN values:",
        np.isnan(normalized).sum()
    )

    # --------------------------------------------
    # Save output
    # --------------------------------------------

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    np.save(
        output_path,
        normalized
    )

    print(
        "Saved:",
        output_path
    )

    return {
    "frames": len(normalized),
    "shape": str(normalized.shape),
    "nan_values": int(
        np.isnan(normalized).sum()
    ),
    "next_timestamp_ms": (
        start_timestamp_ms
        + len(normalized) * 40
        + 1
    )
    }




# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------
    # Load manifest
    # --------------------------------------------

    print(
        "Loading INCLUDE-50 manifest..."
    )

    df = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        "Total manifest entries:",
        len(df)
    )

    # --------------------------------------------
    # Select first 10 videos
    # --------------------------------------------

    test_df = df.copy() if MAX_VIDEOS is None else df.head(MAX_VIDEOS).copy()

    print(
        f"Processing first {len(test_df)} videos."
    )

    # --------------------------------------------
    # Create MediaPipe options
    # --------------------------------------------

    options = HolisticLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),

        running_mode=RunningMode.VIDEO,

        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,

        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,

        min_hand_landmarks_confidence=0.5
    )

    successful = []
    failed = []

    # --------------------------------------------
    # Create ONE MediaPipe instance
    # --------------------------------------------
    #
    # We keep the same landmarker alive while
    # processing all 10 videos.
    #
    # This is important because VIDEO mode
    # requires increasing timestamps.
    # --------------------------------------------

    with HolisticLandmarker.create_from_options(
        options
    ) as landmarker:

        global_timestamp_ms = 0

        for index, row in test_df.iterrows():

            try:

                relative_video = str(
                    row["video_path"]
                )

                split = str(
                    row["split"]
                )

                label = str(
                    row["label"]
                )

                # --------------------------------
                # Input path
                # --------------------------------

                video_path = os.path.normpath(
                    os.path.join(
                        VIDEO_ROOT,
                        relative_video
                    )
                )

                # --------------------------------
                # Unique output filename
                # --------------------------------

                video_name = os.path.splitext(
                    os.path.basename(
                        relative_video
                    )
                )[0]

                # Include parent directory/label
                # to reduce filename collision risk.
                parent = str(
                    row["parent_label"]
                )

                safe_parent = (
                    parent
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("\\", "_")
                )

                safe_label = (
                    label
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(".", "")
                )

                output_filename = (
                    f"{safe_parent}_"
                    f"{safe_label}_"
                    f"{video_name}.npy"
                )

                output_dir = os.path.join(
                    OUTPUT_ROOT,
                    split
                )

                output_path = os.path.join(
                    output_dir,
                    output_filename
                )

                # --------------------------------
                # Skip already processed
                # --------------------------------

                if os.path.exists(
                    output_path
                ):

                    print()
                    print(
                        f"[{len(successful) + len(failed) + 1}/"
                        f"{len(test_df)}] "
                        "Already exists:"
                    )

                    print(
                        output_path
                    )

                    successful.append({
                        "sample_id": video_name,
                        "label": label,
                        "split": split,
                        "video_path": relative_video,
                        "landmark_path": output_path,
                        "status": "existing"
                    })

                    continue

                # --------------------------------
                # Process video
                # --------------------------------

                stats = process_video(
                    video_path,
                    output_path,
                    landmarker,
                    global_timestamp_ms
                )

                global_timestamp_ms = (
                    stats["next_timestamp_ms"]
                )

                successful.append({
                    "sample_id": video_name,
                    "label": label,
                    "split": split,
                    "video_path": relative_video,
                    "landmark_path": output_path,
                    "frames": stats["frames"],
                    "shape": stats["shape"],
                    "nan_values": stats["nan_values"],
                    "status": "processed"
                })

            except Exception as error:

                print()
                print(
                    "FAILED:",
                    row["video_path"]
                )

                print(
                    "Reason:",
                    error
                )

                failed.append({
                    "video_path": row["video_path"],
                    "label": row["label"],
                    "split": row["split"],
                    "error": str(error)
                })

    # --------------------------------------------
    # Save processing report
    # --------------------------------------------

    report_dir = os.path.join(
        OUTPUT_ROOT,
        "reports"
    )

    os.makedirs(
        report_dir,
        exist_ok=True
    )

    successful_df = pd.DataFrame(
        successful
    )

    successful_df.to_csv(
        os.path.join(
            report_dir,
            "processed_10.csv"
        ),
        index=False
    )

    failed_df = pd.DataFrame(
        failed
    )

    failed_df.to_csv(
        os.path.join(
            report_dir,
            "failed_10.csv"
        ),
        index=False
    )

    # --------------------------------------------
    # Final summary
    # --------------------------------------------

    print()
    print("=" * 60)
    print("ALL-VIDEO PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        "Successful:",
        len(successful)
    )

    print(
        "Failed:",
        len(failed)
    )

    print()
    print(
        "Report:",
        os.path.join(
            report_dir,
            "processed_10.csv"
        )
    )


if __name__ == "__main__":
    main()