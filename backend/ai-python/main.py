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


# from fastapi import FastAPI, Request
# from ultralytics import YOLOWorld
# import cv2
# import os
# import torch
# import numpy as np
# from torchvision import models, transforms
# from PIL import Image

# app = FastAPI()
# # V2 model is better for dynamic prompts
# model = YOLOWorld("yolov8s-worldv2.pt")

# # Feature Extractor (Image se video match karne ke liye)
# resnet = models.resnet50(pretrained=True)
# resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
# resnet.eval()

# SAVE_DIR = "detected_frames"
# os.makedirs(SAVE_DIR, exist_ok=True)

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def extract_features(img):
#     img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     img = transform(img).unsqueeze(0)
#     with torch.no_grad():
#         return resnet(img).flatten().numpy()

# @app.post("/process")
# async def process_video(req: Request):
#     data = await req.json()
#     video_path = data.get("filePath")
#     image_path = data.get("imagePath")
#     user_prompt = data.get("prompt", "person")

#     print(f"🚀 Processing: {user_prompt}")

#     # Threshold settings
#     target_threshold = 0.40 
#     # FIX: Ensure prompt is lowercase and clean
#     prompt_list = [p.strip().lower() for p in user_prompt.split(",") if p.strip()]
#     model.set_classes(prompt_list)

#     cap = cv2.VideoCapture(video_path)
#     results_list = []
#     saved_ids = set()
#     frame_id = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
#         frame_id += 1
        
#         # Skip frames for speed
#         if frame_id % 5 != 0: continue 

#         # FIX: Reduced imgsz to 640 for better general detection and speed
#         results = model.track(
#             frame, 
#             conf=target_threshold, 
#             imgsz=640, 
#             persist=True, 
#             verbose=False
#         )

#         if not results[0].boxes or results[0].boxes.id is None: continue

#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         ids = results[0].boxes.id.int().cpu().numpy()
#         confs = results[0].boxes.conf.cpu().numpy()
#         clss = results[0].boxes.cls.int().cpu().numpy()

#         for box, track_id, conf, cls_id in zip(boxes, ids, confs, clss):
#             if track_id in saved_ids: continue

#             label = prompt_list[cls_id]
#             x1, y1, x2, y2 = map(int, box)

#             # Save detection
#             img_name = f"track_{track_id}.jpg"
#             img_path = os.path.join(SAVE_DIR, img_name)
            
#             # Simple crop & save logic
#             cv2.imwrite(img_path, frame[y1:y2, x1:x2])
#             saved_ids.add(track_id)
            
#             results_list.append({
#                 "object": label,
#                 "confidence": float(conf),
#                 "trackingId": int(track_id),
#                 "timestamp": str(frame_id),
#                 "image_path": img_path,
#                 "bbox": [x1, y1, x2, y2]
#             })

#     cap.release()
#     return {"results": results_list}




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

# YOLO World model
model = YOLOWorld("yolov8s-worldv2.pt")

SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# FEATURE EXTRACTOR
# -----------------------------
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])


def extract_features(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        return resnet(img).flatten().numpy()


# -----------------------------
# COLOR DETECTION
# -----------------------------
def get_basic_color(crop):

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    avg_color = crop_rgb.mean(axis=0).mean(axis=0)

    r,g,b = avg_color.astype(int)

    if r > 150 and g < 100 and b < 100:
        return "red"

    if g > 150 and r < 100 and b < 100:
        return "green"

    if b > 150 and r < 100 and g < 100:
        return "blue"

    if r > 200 and g > 200 and b < 100:
        return "yellow"

    if r > 200 and g > 200 and b > 200:
        return "white"

    if r < 80 and g < 80 and b < 80:
        return "black"

    if r > 160 and g > 110 and b > 60:
        return "brown"

    return "unknown"


# -----------------------------
# CORE CLASSES
# -----------------------------
core_classes = [
    "person","man","woman","child",
    "bicycle","motorcycle","bike","scooter",
    "car","truck","bus","van","auto rickshaw",
    "helmet","safety helmet",
    "traffic light","crosswalk","lane",
    "police officer","security guard",
    "person running","person fighting",
    "person falling","person lying on road"
]


# -----------------------------
# VIDEO PROCESS API
# -----------------------------
@app.post("/process")
async def process_video(req: Request):

    data = await req.json()

    video_url = data.get("fileUrl")
    image_url = data.get("imageUrl")
    user_prompt = data.get("prompt","person")

    print("\n🚀 PROCESS STARTED")
    print("Prompt:",user_prompt)

    # -----------------------------
    # DOWNLOAD VIDEO
    # -----------------------------
    video_path = "temp_video.mp4"

    r = requests.get(video_url, stream=True)
    with open(video_path,"wb") as f:
        for chunk in r.iter_content(1024):
            if chunk:
                f.write(chunk)

    # -----------------------------
    # DOWNLOAD IMAGE (OPTIONAL)
    # -----------------------------
    ref_feat = None

    if image_url:

        img_path = "temp_image.jpg"

        r = requests.get(image_url, stream=True)
        with open(img_path,"wb") as f:
            for chunk in r.iter_content(1024):
                if chunk:
                    f.write(chunk)

        ref_img = cv2.imread(img_path)
        ref_feat = extract_features(ref_img)

    # -----------------------------
    # PROMPT HANDLING
    # -----------------------------
    prompt_list = [p.strip().lower() for p in user_prompt.split(",")]

    all_classes = list(set(core_classes + prompt_list))

    model.set_classes(all_classes)

    print("Detection Classes:",all_classes)

    # -----------------------------
    # VIDEO READ
    # -----------------------------
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Total Frames:",total_frames)

    frame_id = 0

    results_list = []
    saved_ids = set()

    target_threshold = 0.35

    # -----------------------------
    # PROCESS FRAMES
    # -----------------------------
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        # frame skipping for speed
        if frame_id % 4 != 0:
            continue

        results = model.track(
            frame,
            conf=target_threshold,
            persist=True,
            imgsz=960,
            verbose=False
        )

        if not results or not results[0].boxes:
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy() if results[0].boxes.id is not None else []
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.int().cpu().numpy()

        for box,track_id,conf,cls_id in zip(boxes,ids,confs,clss):

            if conf < target_threshold:
                continue

            if track_id in saved_ids:
                continue

            label = model.names[cls_id]

            # prompt filtering
            if not any(p in label.lower() for p in prompt_list):
                continue

            x1,y1,x2,y2 = map(int,box)

            crop = frame[max(0,y1):y2, max(0,x1):x2]

            if crop.size == 0:
                continue

            # image matching
            if ref_feat is not None:

                obj_feat = extract_features(crop)

                sim = np.dot(ref_feat,obj_feat) / (
                    np.linalg.norm(ref_feat) * np.linalg.norm(obj_feat)
                )

                if sim < 0.5:
                    continue

                conf = sim

            # color detection
            color = get_basic_color(crop)

            print(f"FOUND {label} {color} ID:{track_id} frame:{frame_id}")

            annotated = frame.copy()

            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(
                annotated,
                f"{label} {color} {conf:.2f}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

            img_path = os.path.join(SAVE_DIR,f"track_{track_id}.jpg")

            cv2.imwrite(img_path,annotated)

            saved_ids.add(track_id)

            results_list.append({
                "object":label,
                "color":color,
                "confidence":float(conf),
                "trackingId":int(track_id),
                "timestamp":frame_id,
                "image_path":img_path,
                "bbox":[x1,y1,x2,y2]
            })

    cap.release()

    print("FINISHED - objects:",len(saved_ids))

    return {
        "results":results_list
    }