import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera could not be opened")
    exit()

print("✅ Camera opened successfully")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("❌ Failed to read camera frame")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

## Press Q to exit it 