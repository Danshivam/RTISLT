import cv2
import mediapipe as mp
import time


# --------------------------------------------------
# MediaPipe setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions

HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


MODEL_PATH = "models/holistic_landmarker.task"


options = HolisticLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,

    min_face_detection_confidence=0.5,
    min_face_landmarks_confidence=0.5,

    min_pose_detection_confidence=0.5,
    min_pose_landmarks_confidence=0.5,

    min_hand_landmarks_confidence=0.5,

    output_face_blendshapes=True
)


# --------------------------------------------------
# Webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# --------------------------------------------------
# Run Holistic Landmarker
# --------------------------------------------------

with HolisticLandmarker.create_from_options(options) as landmarker:

    print("MediaPipe Holistic started.")
    print("Press Q to quit.")

    start_time = time.time()

    while True:

        success, frame = cap.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        # OpenCV gives BGR
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp must increase for VIDEO mode
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Run detection
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # --------------------------------------------------
        # Draw pose
        # --------------------------------------------------

        if result.pose_landmarks:

            for landmark in result.pose_landmarks:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )


        # --------------------------------------------------
        # Draw left hand
        # --------------------------------------------------

        if result.left_hand_landmarks:

            for landmark in result.left_hand_landmarks:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (255, 0, 0),
                    -1
                )


        # --------------------------------------------------
        # Draw right hand
        # --------------------------------------------------

        if result.right_hand_landmarks:

            for landmark in result.right_hand_landmarks:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 0, 255),
                    -1
                )


        # --------------------------------------------------
        # Display
        # --------------------------------------------------

        cv2.imshow(
            "ISL Translator - MediaPipe",
            frame
        )


        # Q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()