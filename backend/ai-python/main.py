




# from fastapi import FastAPI, Request
# from ultralytics import YOLOWorld
# import cv2
# import os

# app = FastAPI()
# model = YOLOWorld("yolov8s-world.pt")

# # 📁 Folder jahan screenshots save honge
# SAVE_DIR = "detected_frames"
# os.makedirs(SAVE_DIR, exist_ok=True)

# @app.post("/process")
# async def process_video(req: Request):
#     data = await req.json()
#     filePath = data.get("filePath")
#     user_prompt = data.get("prompt", "person, car, helmet") 
    
#     custom_classes = [c.strip() for c in user_prompt.split(",")]
#     model.set_classes(custom_classes)

#     cap = cv2.VideoCapture(filePath)
#     results_list = []
#     frame_id = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
#         frame_id += 1
        
#         if frame_id % 15 != 0: continue # Speed ke liye (har 15th frame)

#         results = model.predict(frame, conf=0.65)
        
#         detected_in_this_frame = False
        
#         for r in results:
#             if not r.boxes: continue
            
#             # Frame par boxes draw karne ke liye hum original frame ki copy lenge
#             annotated_frame = frame.copy()

#             for box in r.boxes:
#                 detected_in_this_frame = True
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = float(box.conf[0])
#                 cls_id = int(box.cls[0])
#                 label = custom_classes[cls_id]

#                 # 🎨 Visual Proof: Frame par rectangle draw karo
#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#                 results_list.append({
#                     "object": label,
#                     "confidence": conf,
#                     "trackingId": f"{x1}_{y1}", 
#                     "timestamp": str(frame_id),
#                     "image_path": f"{SAVE_DIR}/frame_{frame_id}.jpg"
#                 })

#             # Agar kuch detect hua, toh image save karo
#             if detected_in_this_frame:
#                 img_name = f"frame_{frame_id}.jpg"
#                 cv2.imwrite(os.path.join(SAVE_DIR, img_name), annotated_frame)
#                 print(f"✅ Detected {label} at frame {frame_id}")

#     cap.release()
#     return {"results": results_list}


from fastapi import FastAPI, Request
from ultralytics import YOLOWorld
import cv2
import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

app = FastAPI()
# V2 model is better for dynamic prompts
model = YOLOWorld("yolov8s-worldv2.pt")

# Feature Extractor (Image se video match karne ke liye)
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        return resnet(img).flatten().numpy()

@app.post("/process")
async def process_video(req: Request):
    data = await req.json()
    video_path = data.get("filePath")
    image_path = data.get("imagePath")
    user_prompt = data.get("prompt", "person")

    print(f"\n🚀 [START] Processing Started...")
    print(f"📂 Video: {os.path.basename(video_path)}")
    print(f"📝 Prompt: {user_prompt}")

    target_threshold = 0.60 
    prompt_list = [p.strip() for p in user_prompt.split(",")]
    model.set_classes(prompt_list)

    ref_feat = None
    if image_path and os.path.exists(image_path):
        print(f"📸 Image provided for matching: {os.path.basename(image_path)}")
        ref_img = cv2.imread(image_path)
        ref_feat = extract_features(ref_img)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞️ Total Frames in Video: {total_frames}")

    results_list = []
    saved_ids = set()
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        
        # 🏎️ SPEED BOOST: Har 5th frame skip ki jagah, hum logic ko fast karenge
        if frame_id % 5 != 0: continue 

        # 🎯 OPTIMIZATION: imgsz=640 and half=True (Speed optimized)
        # Device automatically handles CPU/GPU
        results = model.track(
            frame, 
            conf=target_threshold, 
            imgsz=640, # 1280 se 640 kiya for 2x speed
            persist=True, 
            verbose=False,
            half=True if torch.cuda.is_available() else False 
        )

        if frame_id % 50 == 0:
            print(f"⏳ Progress: {frame_id}/{total_frames} frames processed...")

        if not results[0].boxes or results[0].boxes.id is None: continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.int().cpu().numpy()

        for box, track_id, conf, cls_id in zip(boxes, ids, confs, clss):
            if conf < target_threshold: continue
            if track_id in saved_ids: continue

            x1, y1, x2, y2 = map(int, box)
            label = prompt_list[cls_id]

            # Image Matching Logic
            if ref_feat is not None:
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0: continue
                obj_feat = extract_features(crop)
                sim = np.dot(ref_feat, obj_feat) / (np.linalg.norm(ref_feat) * np.linalg.norm(obj_feat))
                if sim < target_threshold: continue
                conf = sim 

            print(f"✅ Found: {label} (ID: {track_id}) at Frame {frame_id}")

            img_path = os.path.join(SAVE_DIR, f"track_{track_id}.jpg")
            annotated = frame.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imwrite(img_path, annotated)
            saved_ids.add(track_id)
            
            results_list.append({
                "object": label,
                "confidence": float(conf),
                "trackingId": int(track_id),
                "timestamp": str(frame_id),
                "image_path": img_path,
                "bbox": [x1, y1, x2, y2]
            })

    cap.release()
    print(f"🏁 [FINISHED] Total unique objects detected: {len(saved_ids)}")
    return {"results": results_list}