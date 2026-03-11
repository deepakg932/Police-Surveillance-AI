import os
import cv2
import requests
import numpy as np
from fastapi import FastAPI, Request
from ultralytics import YOLOWorld

app = FastAPI()
model = YOLOWorld("yolov8m-worldv2.pt")

SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# Helper: Do cheezon ke beech overlap check karne ke liye (IOU logic)
def is_overlapping(box1, box2):
    x1, y1, x2, y2 = box1
    head_zone = [x1, y1, x2, y1 + (y2 - y1) * 0.4] # Rider ka sar wala area
    
    hx1, hy1, hx2, hy2 = box2
    xA, yA = max(head_zone[0], hx1), max(head_zone[1], hy1)
    xB, yB = min(head_zone[2], hx2), min(head_zone[3], hy2)
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea > 0

@app.post("/process")
async def process_video(req: Request):
    try:
        data = await req.json()
        video_url = data.get("fileUrl")
        user_prompt = data.get("prompt", "").lower().strip()

        # 🎯 1. SMART INTENT CHECK
        # Agar prompt mein 'without' ya 'no' hai, toh logic switch hoga
        is_negative_query = "without" in user_prompt or "no " in user_prompt or "bina" in user_prompt

        if is_negative_query:
            # Model ko bolo dono dhoonde (Rider aur Helmet)
            model.set_classes(["motorcyclist", "helmet"])
        else:
            # Normal Dynamic Mode
            model.set_classes([user_prompt])

        # Video Download
        video_path = "temp_video.mp4"
        r = requests.get(video_url, stream=True)
        with open(video_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: f.write(chunk)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        results_list = []
        saved_ids = set()
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_id += 1
            if frame_id % 6 != 0: continue 

            results = model.predict(frame, conf=0.35, imgsz=640, verbose=False)
            if not results[0].boxes: continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            # 🎯 2. ACCURACY LOGIC
            if is_negative_query:
                # Violation Detection: Rider dhoondo jiske sar par helmet NA HO
                bikers = [boxes[i] for i, c in enumerate(clss) if c == 0]
                helmets = [boxes[i] for i, c in enumerate(clss) if c == 1]
                
                for b_box in bikers:
                    has_helmet = any(is_overlapping(b_box, h_box) for h_box in helmets)
                    if not has_helmet:
                        save_and_report(frame, b_box, user_prompt, frame_id, fps, results_list, saved_ids)
            else:
                # Normal Dynamic Detection
                for i, box in enumerate(boxes):
                    save_and_report(frame, box, user_prompt, frame_id, fps, results_list, saved_ids)

            if len(results_list) >= 15: break

        cap.release()
        return {"status": "success", "results": results_list}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def save_and_report(frame, box, label, frame_id, fps, results_list, saved_ids):
    track_id = f"det_{int(box[0]/10)}_{frame_id}"
    if track_id not in saved_ids:
        x1, y1, x2, y2 = map(int, box)
        img_path = os.path.join(SAVE_DIR, f"match_{frame_id}.jpg")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red box for violation
        cv2.putText(frame, label.upper(), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(img_path, frame)
        saved_ids.add(track_id)
        results_list.append({"object": label, "image_path": img_path, "timestamp": f"{frame_id/fps:.2f}s"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)