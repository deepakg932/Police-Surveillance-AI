




# # from fastapi import FastAPI, Request
# # from ultralytics import YOLOWorld
# # import cv2
# # import os

# # app = FastAPI()
# # model = YOLOWorld("yolov8s-world.pt")

# # # 📁 Folder jahan screenshots save honge
# # SAVE_DIR = "detected_frames"
# # os.makedirs(SAVE_DIR, exist_ok=True)

# # @app.post("/process")
# # async def process_video(req: Request):
# #     data = await req.json()
# #     filePath = data.get("filePath")
# #     user_prompt = data.get("prompt", "person, car, helmet") 
    
# #     custom_classes = [c.strip() for c in user_prompt.split(",")]
# #     model.set_classes(custom_classes)

# #     cap = cv2.VideoCapture(filePath)
# #     results_list = []
# #     frame_id = 0

# #     while cap.isOpened():
# #         ret, frame = cap.read()
# #         if not ret: break
# #         frame_id += 1
        
# #         if frame_id % 15 != 0: continue # Speed ke liye (har 15th frame)

# #         results = model.predict(frame, conf=0.65)
        
# #         detected_in_this_frame = False
        
# #         for r in results:
# #             if not r.boxes: continue
            
# #             # Frame par boxes draw karne ke liye hum original frame ki copy lenge
# #             annotated_frame = frame.copy()

# #             for box in r.boxes:
# #                 detected_in_this_frame = True
# #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
# #                 conf = float(box.conf[0])
# #                 cls_id = int(box.cls[0])
# #                 label = custom_classes[cls_id]

# #                 # 🎨 Visual Proof: Frame par rectangle draw karo
# #                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #                 cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
# #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# #                 results_list.append({
# #                     "object": label,
# #                     "confidence": conf,
# #                     "trackingId": f"{x1}_{y1}", 
# #                     "timestamp": str(frame_id),
# #                     "image_path": f"{SAVE_DIR}/frame_{frame_id}.jpg"
# #                 })

# #             # Agar kuch detect hua, toh image save karo
# #             if detected_in_this_frame:
# #                 img_name = f"frame_{frame_id}.jpg"
# #                 cv2.imwrite(os.path.join(SAVE_DIR, img_name), annotated_frame)
# #                 print(f"✅ Detected {label} at frame {frame_id}")

# #     cap.release()
# #     return {"results": results_list}


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

#     print(f"\n🚀 [START] Processing Started...")
#     print(f"📂 Video: {os.path.basename(video_path)}")
#     print(f"📝 Prompt: {user_prompt}")

#     target_threshold = 0.35
#     prompt_list = [p.strip() for p in user_prompt.split(",")]
#     model.set_classes(prompt_list)

#     ref_feat = None
#     if image_path and os.path.exists(image_path):
#         print(f"📸 Image provided for matching: {os.path.basename(image_path)}")
#         ref_img = cv2.imread(image_path)
#         ref_feat = extract_features(ref_img)

#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"🎞️ Total Frames in Video: {total_frames}")

#     results_list = []
#     saved_ids = set()
#     frame_id = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
#         frame_id += 1
        
#         # 🏎️ SPEED BOOST: Har 5th frame skip ki jagah, hum logic ko fast karenge
#         if frame_id % 5 != 0: continue 

#         # 🎯 OPTIMIZATION: imgsz=640 and half=True (Speed optimized)
#         # Device automatically handles CPU/GPU
#         results = model.track(
#             frame, 
#             conf=target_threshold, 
#             imgsz=1280, # 1280 se 640 kiya for 2x speed
#             persist=True, 
#             verbose=False,
#             half=True if torch.cuda.is_available() else False 
#         )

#         if frame_id % 50 == 0:
#             print(f"⏳ Progress: {frame_id}/{total_frames} frames processed...")

#         if not results[0].boxes or results[0].boxes.id is None: continue

#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         ids = results[0].boxes.id.int().cpu().numpy()
#         confs = results[0].boxes.conf.cpu().numpy()
#         clss = results[0].boxes.cls.int().cpu().numpy()

#         for box, track_id, conf, cls_id in zip(boxes, ids, confs, clss):
#             if conf < target_threshold: continue
#             if track_id in saved_ids: continue

#             x1, y1, x2, y2 = map(int, box)
#             label = prompt_list[cls_id]

#             # Image Matching Logic
#             if ref_feat is not None:
#                 crop = frame[max(0, y1):y2, max(0, x1):x2]
#                 if crop.size == 0: continue
#                 obj_feat = extract_features(crop)
#                 sim = np.dot(ref_feat, obj_feat) / (np.linalg.norm(ref_feat) * np.linalg.norm(obj_feat))
#                 if sim < target_threshold: continue
#                 conf = sim 

#             print(f"✅ Found: {label} (ID: {track_id}) at Frame {frame_id}")

#             img_path = os.path.join(SAVE_DIR, f"track_{track_id}.jpg")
#             annotated = frame.copy()
#             cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1-10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
#             cv2.imwrite(img_path, annotated)
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
#     print(f"🏁 [FINISHED] Total unique objects detected: {len(saved_ids)}")
#     return {"results": results_list}







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

#     print(f"\n🚀 [START] Processing Started...")
#     print(f"📂 Video: {os.path.basename(video_path)}")
#     print(f"📝 Prompt: {user_prompt}")

#     target_threshold = 0.60
#     prompt_list = [p.strip() for p in user_prompt.split(",")]
#     model.set_classes(prompt_list)

#     ref_feat = None
#     if image_path and os.path.exists(image_path):
#         print(f"📸 Image provided for matching: {os.path.basename(image_path)}")
#         ref_img = cv2.imread(image_path)
#         ref_feat = extract_features(ref_img)

#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"🎞️ Total Frames in Video: {total_frames}")

#     results_list = []
#     saved_ids = set()
#     frame_id = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
#         frame_id += 1
       
#         # 🏎️ SPEED BOOST: Har 5th frame skip ki jagah, hum logic ko fast karenge
#         if frame_id % 5 != 0: continue

#         # 🎯 OPTIMIZATION: imgsz=640 and half=True (Speed optimized)
#         # Device automatically handles CPU/GPU
#         results = model.track(
#             frame,
#             conf=target_threshold,
#             imgsz=640, # 1280 se 640 kiya for 2x speed
#             persist=True,
#             verbose=False,
#             half=True if torch.cuda.is_available() else False
#         )

#         if frame_id % 50 == 0:
#             print(f"⏳ Progress: {frame_id}/{total_frames} frames processed...")

#         if not results[0].boxes or results[0].boxes.id is None: continue

#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         ids = results[0].boxes.id.int().cpu().numpy()
#         confs = results[0].boxes.conf.cpu().numpy()
#         clss = results[0].boxes.cls.int().cpu().numpy()

#         for box, track_id, conf, cls_id in zip(boxes, ids, confs, clss):
#             if conf < target_threshold: continue
#             if track_id in saved_ids: continue

#             x1, y1, x2, y2 = map(int, box)
#             label = prompt_list[cls_id]

#             # Image Matching Logic
#             if ref_feat is not None:
#                 crop = frame[max(0, y1):y2, max(0, x1):x2]
#                 if crop.size == 0: continue
#                 obj_feat = extract_features(crop)
#                 sim = np.dot(ref_feat, obj_feat) / (np.linalg.norm(ref_feat) * np.linalg.norm(obj_feat))
#                 if sim < target_threshold: continue
#                 conf = sim

#             print(f"✅ Found: {label} (ID: {track_id}) at Frame {frame_id}")

#             img_path = os.path.join(SAVE_DIR, f"track_{track_id}.jpg")
#             annotated = frame.copy()
#             cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
           
#             cv2.imwrite(img_path, annotated)
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
#     print(f"🏁 [FINISHED] Total unique objects detected: {len(saved_ids)}")
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

# # @app.post("/process")
# # async def process_video(req: Request):
# #     data = await req.json()
# #     video_path = data.get("filePath")
# #     image_path = data.get("imagePath")
# #     user_prompt = data.get("prompt", "person")

# #     print(f"\n🚀 [START] Processing Started...")
# #     print(f"📂 Video: {os.path.basename(video_path)}")
# #     print(f"📝 Prompt: {user_prompt}")

# #     target_threshold = 0.35
# #     prompt_list = [p.strip() for p in user_prompt.split(",")]
# #     model.set_classes(prompt_list)

# #     ref_feat = None
# #     if image_path and os.path.exists(image_path):
# #         print(f"📸 Image provided for matching: {os.path.basename(image_path)}")
# #         ref_img = cv2.imread(image_path)
# #         ref_feat = extract_features(ref_img)

# #     cap = cv2.VideoCapture(video_path)
# #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #     print(f"🎞️ Total Frames in Video: {total_frames}")

# #     results_list = []
# #     saved_ids = set()
# #     frame_id = 0

# #     while cap.isOpened():
# #         ret, frame = cap.read()
# #         if not ret: break
# #         frame_id += 1
       
# #         # 🏎️ SPEED BOOST: Har 5th frame skip ki jagah, hum logic ko fast karenge
# #         if frame_id % 5 != 0: continue

# #         # 🎯 OPTIMIZATION: imgsz=640 and half=True (Speed optimized)
# #         # Device automatically handles CPU/GPU
# #         results = model.track(
# #             frame,
# #             conf=target_threshold,
# #             imgsz=1280, # 1280 se 640 kiya for 2x speed
# #             persist=True,
# #             verbose=False,
# #             half=True if torch.cuda.is_available() else False
# #         )

# #         if frame_id % 50 == 0:
# #             print(f"⏳ Progress: {frame_id}/{total_frames} frames processed...")

# #         if not results[0].boxes or results[0].boxes.id is None: continue

# #         boxes = results[0].boxes.xyxy.cpu().numpy()
# #         ids = results[0].boxes.id.int().cpu().numpy()
# #         confs = results[0].boxes.conf.cpu().numpy()
# #         clss = results[0].boxes.cls.int().cpu().numpy()

# #         for box, track_id, conf, cls_id in zip(boxes, ids, confs, clss):
# #             if conf < target_threshold: continue
# #             if track_id in saved_ids: continue

# #             x1, y1, x2, y2 = map(int, box)
# #             label = prompt_list[cls_id]

# #             # Image Matching Logic
# #             if ref_feat is not None:
# #                 crop = frame[max(0, y1):y2, max(0, x1):x2]
# #                 if crop.size == 0: continue
# #                 obj_feat = extract_features(crop)
# #                 sim = np.dot(ref_feat, obj_feat) / (np.linalg.norm(ref_feat) * np.linalg.norm(obj_feat))
# #                 if sim < target_threshold: continue
# #                 conf = sim

# #             print(f"✅ Found: {label} (ID: {track_id}) at Frame {frame_id}")

# #             img_path = os.path.join(SAVE_DIR, f"track_{track_id}.jpg")
# #             annotated = frame.copy()
# #             cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #             cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1-10),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
           
# #             cv2.imwrite(img_path, annotated)
# #             saved_ids.add(track_id)
           
# #             results_list.append({
# #                 "object": label,
# #                 "confidence": float(conf),
# #                 "trackingId": int(track_id),
# #                 "timestamp": str(frame_id),
# #                 "image_path": img_path,
# #                 "bbox": [x1, y1, x2, y2]
# #             })

# #     cap.release()
# #     print(f"🏁 [FINISHED] Total unique objects detected: {len(saved_ids)}")
# #     return {"results": results_list}

# @app.post("/process")
# async def process_video(req: Request):
#     data = await req.json()
#     video_path = data.get("filePath")
#     image_path = data.get("imagePath")
#     user_prompt = data.get("prompt", "person, helmet")

#     # Prompt list ko saaf karein
#     prompt_list = [p.strip() for p in user_prompt.split(",")]
#     model.set_classes(prompt_list)

#     cap = cv2.VideoCapture(video_path)
#     results_list = []
#     saved_ids = set()
#     frame_id = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
#         frame_id += 1
        
#         # Speed ke liye har 5th frame
#         if frame_id % 5 != 0: continue

#         results = model.track(
#             frame,
#             conf=0.25, # Base confidence thoda kam rakha taaki tracking na toote
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
#             # 🎯 60% (0.60) MATCH CHECK: Agar isse kam hai toh ignore karo
#             if conf < 0.60: continue

#             label = prompt_list[cls_id]
#             x1, y1, x2, y2 = map(int, box)
#             img_path = ""

#             # Pehli baar detection hone par annotated image save karo
#             if track_id not in saved_ids:
#                 img_path = os.path.join(SAVE_DIR, f"track_{track_id}.jpg")
                
#                 # Frame ki copy banao taaki box draw kar sakein
#                 annotated_frame = frame.copy()
                
#                 # 🟩 DRAW GREEN BOX: (BGR format mein (0, 255, 0) green hota hai)
#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
#                 # 📝 DRAW TEXT: Object Name + Confidence %
#                 label_text = f"{label} {int(conf * 100)}%"
#                 cv2.putText(annotated_frame, label_text, (x1, y1 - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#                 # Padding add karke crop karo taaki box pura dikhe
#                 margin = 25
#                 crop = annotated_frame[max(0, y1-margin):y2+margin, max(0, x1-margin):x2+margin]
                
#                 if crop.size > 0:
#                     cv2.imwrite(img_path, crop)
#                 else:
#                     cv2.imwrite(img_path, annotated_frame)
                
#                 saved_ids.add(track_id)

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

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from ultralytics import YOLOWorld
import cv2
import os
import torch
import numpy as np
import requests
from torchvision import models, transforms
from PIL import Image

app = FastAPI()

# 🚀 Model loading (V2 is better for dynamic prompts)
model = YOLOWorld("yolov8s-worldv2.pt")

# 📸 Feature Extractor for image matching (ResNet50)
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# Image preprocessing for ResNet
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
    
    # 🎯 Node.js se ye URLs aur data aayenge
    video_url = data.get("fileUrl") or data.get("videoUrl")
    image_url = data.get("imageUrl") or data.get("imagePath")
    user_prompt = data.get("prompt", "person")

    print(f"\n🚀 [START] Processing Task...")
    print(f"🔗 Video URL: {video_url}")
    print(f"📝 Prompt: {user_prompt}")

    if not video_url:
        raise HTTPException(status_code=400, detail="Video URL missing")

    # 1️⃣ Video Download Logic
    video_local_path = "temp_video.mp4"
    try:
        r = requests.get(video_url, stream=True, timeout=10)
        with open(video_local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video download failed: {e}")

    # 2️⃣ Image Download Logic (Matching ke liye)
    image_local_path = None
    if image_url and image_url.startswith("http"):
        image_local_path = "temp_match.jpg"
        r = requests.get(image_url, stream=True)
        with open(image_local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: f.write(chunk)
    elif image_url:
        image_local_path = image_url # Local path if provided

    # 3️⃣ YOLO Setup
    prompt_list = [p.strip() for p in user_prompt.split(",")]
    model.set_classes(prompt_list)

    # Threshold logic
    prompt_lower = user_prompt.lower()
    target_threshold = 0.55 if any(x in prompt_lower for x in ["helmet", "wearing"]) else 0.50

    # Extract reference features if image exists
    ref_feat = None
    if image_local_path and os.path.exists(image_local_path):
        ref_img = cv2.imread(image_local_path)
        if ref_img is not None:
            ref_feat = extract_features(ref_img)

    # 4️⃣ Video Processing Loop
    cap = cv2.VideoCapture(video_local_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Frame skipping logic for speed
    frame_interval = 3 if (total_frames / fps) < 15 else 6
    
    results_list = []
    saved_ids = set()
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1

        if frame_id % frame_interval != 0: continue 

        # 🎯 Inference with Speed Optimization
        results = model.track(
            frame, 
            conf=target_threshold, 
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
            if track_id in saved_ids: continue

            x1, y1, x2, y2 = map(int, box)
            label = prompt_list[cls_id]

            # 🧩 Image Matching (Similarity) logic
            if ref_feat is not None:
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0: continue
                obj_feat = extract_features(crop)
                sim = np.dot(ref_feat, obj_feat) / (np.linalg.norm(ref_feat) * np.linalg.norm(obj_feat))
                if sim < 0.45: continue # Matching threshold
                conf = sim 

            print(f"✅ Found: {label} (ID: {track_id})")

            # Save Annotated Image
            img_filename = f"track_{track_id}.jpg"
            img_path = os.path.join(SAVE_DIR, img_filename)
            
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
                "timestamp": str(round(frame_id / fps, 2)), # Seconds mein conversion
                "image_path": img_path,
                "bbox": [x1, y1, x2, y2]
            })

    cap.release()
    print(f"🏁 [FINISHED] Detected: {len(saved_ids)} objects")
    
    return {
        "videoUrl": video_url, 
        "results": results_list,
        "totalDetected": len(saved_ids)
    }

# ============ MULTIMODAL AI SECTION ============

class AskRequest(BaseModel):
    videoPath: str
    videoUrl: str = "No URL Provided"
    prompt: str

@app.post("/ask")
async def ask_video(req: AskRequest):
    # Agar local path nahi hai toh download logic yahan bhi dal sakte ho
    if not os.path.exists(req.videoPath):
        raise HTTPException(status_code=404, detail="Video file not found locally")

    try:
        from services.video_processor import extract_frames
        from services.multimodal_client import ask_gemini
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Service imports missing: {e}")

    frames = extract_frames(req.videoPath, max_frames=12)
    if not frames:
        raise HTTPException(status_code=400, detail="Could not extract frames")

    answer = ask_gemini(frames, req.prompt)
    return {
        "answer": answer, 
        "videoUrl": req.videoUrl, 
        "frameCount": len(frames)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)