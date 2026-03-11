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

def is_contained(box_inner, box_outer):
    ix1, iy1, ix2, iy2 = box_inner
    ox1, oy1, ox2, oy2 = box_outer
    xA = max(ix1, ox1)
    yA = max(iy1, oy1)
    xB = min(ix2, ox2)
    yB = min(iy2, oy2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    innerArea = (ix2 - ix1) * (iy2 - iy1)
    return interArea / float(innerArea) > 0.5

@app.post("/process")
async def process_video(req: Request):
    try:
        data = await req.json()
        video_url = data.get("fileUrl")
        user_prompt = data.get("prompt", "").lower().strip()

        print(f"\n🚀 Starting Processing...")
        print(f"📂 Video URL: {video_url}")
        print(f"📝 Prompt: {user_prompt}")

        core_classes = ["person", "motorcycle", "scooter", "helmet", "car"]
        model.set_classes(core_classes + [user_prompt])

        video_path = "temp_video.mp4"
        print(f"⏳ Downloading video...")
        r = requests.get(video_url, stream=True)
        with open(video_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: f.write(chunk)
        print(f"✅ Download Complete.")

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

            results = model.predict(frame, conf=0.45, imgsz=640, verbose=False)
            if not results[0].boxes: continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            names = model.names

            persons, vehicles, helmets = [], [], []

            for i, box in enumerate(boxes):
                label = names[clss[i]]
                if label == "person": persons.append((box, confs[i]))
                elif label in ["motorcycle", "scooter"]: vehicles.append(box)
                elif label == "helmet": helmets.append(box)

            # Debugging Print: Har 10th processed frame ka status
            if frame_id % 30 == 0:
                print(f"📸 Frame {frame_id}: Found {len(persons)} persons, {len(vehicles)} bikes, {len(helmets)} helmets")

            for p_box, p_conf in persons:
                # Logic: Check if person is on bike
                on_bike = any(is_contained(p_box, v_box) for v_box in vehicles)
                
                if on_bike:
                    has_helmet = any(is_contained(h_box, p_box) for h_box in helmets)
                    is_violation = "without" in user_prompt or "no" in user_prompt
                    
                    if is_violation and not has_helmet:
                        print(f"⚠️ VIOLATION DETECTED: Rider without helmet at {frame_id/fps:.2f}s")
                        save_and_report(frame, p_box, "Rider WITHOUT Helmet", p_conf, frame_id, fps, results_list, saved_ids)
                    elif not is_violation and has_helmet:
                        print(f"✅ SAFE RIDER DETECTED: Rider with helmet at {frame_id/fps:.2f}s")
                        save_and_report(frame, p_box, "Rider WITH Helmet", p_conf, frame_id, fps, results_list, saved_ids)

            if len(results_list) >= 15: 
                print(f"🛑 Limit reached (15 detections). Stopping.")
                break

        cap.release()
        if os.path.exists(video_path): os.remove(video_path)
        
        print(f"🏁 Processing finished. Total Detections: {len(results_list)}")
        return {"status": "success", "results": results_list}

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {"status": "error", "message": str(e)}

def save_and_report(frame, box, label, conf, frame_id, fps, results_list, saved_ids):
    track_id = f"det_{int(box[0]/10)}_{frame_id}"
    
    if track_id not in saved_ids:
        x1, y1, x2, y2 = map(int, box)
        img_name = f"res_{frame_id}.jpg"
        img_path = os.path.join(SAVE_DIR, img_name)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(img_path, frame)
        
        saved_ids.add(track_id)
        
        results_list.append({
            "object": label,
            "confidence": float(conf),
            "image_path": img_path,
            "timestamp": f"{frame_id/fps:.2f}s",
            "trackingId": track_id,
            "bbox": [x1, y1, x2, y2]
        })

if __name__ == "__main__":
    import uvicorn
    print("🛰️ Starting Server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)