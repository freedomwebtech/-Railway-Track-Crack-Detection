import cv2
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# === Set your Gemini API key ===
GOOGLE_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)

# === Load YOLO crack detection model ===
model = YOLO("best.pt")  # Replace with your crack detection YOLO model
names = model.names

# === Video source ===
cap = cv2.VideoCapture("track.mp4")  # Use 0 for webcam

# === Utilities ===
last_sent_times = {}       # Track ID → last sent time
track_results = {}         # Track ID → Gemini response

def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

def analyze_crack_with_gemini(base64_image, track_id):
    try:
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Analyze this railway track image and give only the following information in short:\n\n"
                                         "**Crack Intensity:** (Low / Medium / High)\n"
                                         "**Recommended Maintenance:** (e.g., Welding, Track Replacement, Minor Inspection)"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
        response = gemini_model.invoke([message])
        result = response.content.strip()
        print(f"[Track {track_id}] Gemini Response:\n{result}")
        track_results[track_id] = result
    except Exception as e:
        print(f"[Track {track_id}] Error:", e)

# === Main Loop ===
cv2.namedWindow("Railway Crack Detection")
frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))

    # Run YOLO detection with tracking
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            name = names[class_id].lower()
            if "crack" not in name:
                continue

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Base label
            cvzone.putTextRect(frame, f"Crack {track_id}", (x1, y1 - 20), scale=1, thickness=1, offset=5)

            # Show Gemini info in multi-line numbered format
            if track_id in track_results:
                gemini_text = track_results[track_id]
                lines = gemini_text.strip().split("\n")
                for i, line in enumerate(lines):
                    numbered_line = f"{i+1}. {line.strip()}"
                    # Position each line above the previous
                    cvzone.putTextRect(frame, numbered_line, (x1, y1 - 45 - i * 25), scale=0.9, thickness=1, offset=4)

            # Send to Gemini every 5 seconds
            current_time = time.time()
            last_time = last_sent_times.get(track_id, 0)
            if current_time - last_time > 5:
                last_sent_times[track_id] = current_time
                crop = frame[y1:y2, x1:x2]
                base64_img = encode_image_to_base64(crop)
                threading.Thread(target=analyze_crack_with_gemini, args=(base64_img, track_id)).start()

    # Show the frame
    cv2.imshow("Railway Crack Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
