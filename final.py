import cv2
from ultralytics import YOLO
import cvzone

# Load YOLOv8 model (pre-trained on COCO)
model = YOLO('best.pt')
names = model.names  # Class name mapping

# Video source
cap = cv2.VideoCapture("track.mp4")  # Change to 0 for webcam



# Tracking history
hist = {}

# Counts

out_count = 0
frame_count = 0

# Mouse callback for pixel debug (optional)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

line_y=263
# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue  # Skip frames for faster processing

    # Resize frame
    frame = cv2.resize(frame, (1020, 600))

    # Run YOLOv8 tracking (only on car=2, motorcycle=3)
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            name = names[class_id]

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{name}', (x1, y1), 1, 1)

    
#    cvzone.putTextRect(frame, f'OUT: {out_count}', (40, 100), scale=2, thickness=2,
#                       colorT=(255, 255, 255), colorR=(0, 0, 255))

    # Draw counting line
#    cv2.line(frame, (0,line_y), (frame.shape[1],line_y), (255, 0, 0), 2)




    # Show frame
    cv2.imshow("RGB", frame)

    # Exit on ESC key
    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
