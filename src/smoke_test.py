from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # downloads automatically the first time
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    results = model(frame, verbose=False)
    cv2.imshow("smoke test", results[0].plot())#imshow a showing window on the desktop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
