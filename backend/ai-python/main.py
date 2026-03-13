from fastapi import FastAPI, Request
from ultralytics import YOLOWorld
import cv2
import os
import torch
import numpy as np
import requests
from torchvision import models, transforms
from PIL import Image

app = FastAPI()

# YOLO-World v2 is excellent for zero-shot (any prompt)
model = YOLOWorld("yolov8s-worldv2.pt")

# Feature Extractor for Image-to-Video matching
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

    video_url = data.get("fileUrl")
    image_url = data.get("imageUrl")
    user_prompt = data.get("prompt", "person")

    print(f"\n🚀 [START] Processing: {user_prompt}")

    # 1. SET DYNAMIC CLASSES
    # This allows YOLO-World to prepare for your specific prompt
    prompt_list = [p.strip() for p in user_prompt.split(",")]
    model.set_classes(prompt_list)

    # 2. DOWNLOAD VIDEO
    video_path = "temp_video.mp4"
    if video_url:
        r = requests.get(video_url, stream=True)
        with open(video_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: f.write(chunk)

    # 3. DOWNLOAD REFERENCE IMAGE
    image_path = None
    if image_url:
        image_path = "temp_image.jpg"
        r = requests.get(image_url, stream=True)
        with open(image_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: f.write(chunk)

    ref_feat = None
    if image_path and os.path.exists(image_path):
        ref_img = cv2.imread(image_path)
        if ref_img is not None:
            ref_feat = extract_features(ref_img)

    # 4. DETECTION SETTINGS
    # Increased threshold to 0.55 to stop false positives seen in your images
    target_threshold = 0.55 
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results_list = []
    saved_ids = set()
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        
        # Skip frames for processing speed
        if frame_id % 5 != 0: continue 

        # Running Inference
        results = model.track(
            frame, 
            conf=0.25, # Lower internal conf to let tracker maintain IDs
            imgsz=640, 
            persist=True, 
            verbose=False,
            half=True if torch.cuda.is_available() else False 
        )

        if not results[0].boxes or results[0].boxes.id is None: continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.int().cpu().numpy()

        for box, track_id, conf, cls_id in zip(boxes, ids, confs, clss):
            # FILTER 1: Confidence Check
            # Your images showed detections at 0.40; this will block them.
            if conf < target_threshold: 
                continue

            if track_id in saved_ids: 
                continue

            x1, y1, x2, y2 = map(int, box)
            label = prompt_list[cls_id]

            # FILTER 2: Visual Similarity (If image provided)
            if ref_feat is not None:
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0: continue
                obj_feat = extract_features(crop)
                sim = np.dot(ref_feat, obj_feat) / (np.linalg.norm(ref_feat) * np.linalg.norm(obj_feat))
                
                # If similarity is low, it's not the person from the photo
                if sim < 0.65: continue
                conf = sim 

            # Save Detection
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
    print(f"🏁 Finished. Detected {len(saved_ids)} unique objects.")
    return {"results": results_list}