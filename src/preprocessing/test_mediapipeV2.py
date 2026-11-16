
import cv2
import mediapipe as mp
import time

# ------------------------------------------
# MediaPipe Holistic setup
# ------------------------------------------

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------------------
# Webcam setup
# ------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    holistic.close()
    raise SystemExit

print("MediaPipe Holistic started!")
print("Press Q to quit.")

prev_time = 0

# ------------------------------------------
# Main loop
# ------------------------------------------

try:
    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Unable to read webcam frame.")
            break

        # Mirror webcam for natural interaction
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb_frame.flags.writeable = False

        # Run Holistic detection
        results = holistic.process(rgb_frame)

        rgb_frame.flags.writeable = True

        # ----------------------------------
        # Draw pose landmarks
        # ----------------------------------

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        # ----------------------------------
        # Draw face landmarks
        # ----------------------------------

        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=(
                    mp_drawing_styles
                    .get_default_face_mesh_contours_style()
                )
            )

        # ----------------------------------
        # Draw left hand
        # ----------------------------------

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        # ----------------------------------
        # Draw right hand
        # ----------------------------------

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        # ----------------------------------
        # FPS counter
        # ----------------------------------

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ----------------------------------
        # Display
        # ----------------------------------

        cv2.imshow(
            "RTISLT - MediaPipe Holistic",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    holistic.close()
    cv2.destroyAllWindows()
    print("MediaPipe stopped.")
