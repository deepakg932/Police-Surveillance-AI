# import os
# import warnings

# # 1. SUPPRESS ALL NOISY WARNINGS
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# from fastapi import FastAPI, Request
# import cv2
# import torch
# import requests
# import numpy as np
# from groundingdino.util.inference import load_model, predict
# import groundingdino.datasets.transforms as T
# from ultralytics import YOLO
# from PIL import Image
# import open_clip

# # Force CPU mode as requested
# DEVICE = "cpu"

# app = FastAPI()

# # -----------------------------
# # LOAD MODELS ONCE (Startup)
# # -----------------------------
# print("Initializing Models... please wait.")
# ground_model = load_model(
#     "groundingdino/config/GroundingDINO_SwinT_OGC.py", 
#     "groundingdino_swint_ogc.pth"
# )
# ground_model.to(DEVICE)

# # CLIP for verifying color (e.g., "RED")
# clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
# clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
# clip_model.to(DEVICE).eval()

# SAVE_DIR = "detected_frames"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # -----------------------------
# # HELPER FUNCTIONS
# # -----------------------------
# def run_groundingdino(frame, prompt):
#     """Detects objects based on text prompt and returns pixel coordinates."""
#     transform = T.Compose([
#         T.RandomResize([800], max_size=1333),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])
    
#     image_source = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     image_transformed, _ = transform(image_source, None)

#     with torch.no_grad():
#         boxes, logits, phrases = predict(
#             model=ground_model,
#             image=image_transformed,
#             caption=prompt.lower(),
#             box_threshold=0.35,
#             text_threshold=0.25,
#             device=DEVICE
#         )
    
#     h, w, _ = frame.shape
#     pixel_boxes = []
#     for box in boxes:
#         # Convert from normalized cx, cy, w, h to pixel x1, y1, x2, y2
#         box = box * torch.Tensor([w, h, w, h])
#         cx, cy, bw, bh = box.tolist()
#         x1, y1 = int(cx - bw/2), int(cy - bh/2)
#         x2, y2 = int(x1 + bw), int(y1 + bh)
#         pixel_boxes.append([max(0, x1), max(0, y1), min(w, x2), min(h, y2)])
        
#     return pixel_boxes

# def verify_with_clip(image_crop, prompt):
#     """Uses CLIP to double-check if the crop matches the prompt (e.g. 'RED CAR')."""
#     image_pil = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
#     image_input = clip_preprocess(image_pil).unsqueeze(0).to(DEVICE)
#     text_input = clip_tokenizer([prompt]).to(DEVICE)
    
#     with torch.no_grad():
#         image_features = clip_model.encode_image(image_input)
#         text_features = clip_model.encode_text(text_input)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#         similarity = (image_features @ text_features.T).item()
#     return similarity

# # -----------------------------
# # API ENDPOINT
# # -----------------------------
# @app.post("/process")
# async def process_video(req: Request):
#     data = await req.json()
#     video_url = data.get("fileUrl")
#     prompt = data.get("prompt", "car").upper()
    
#     print(f"--- STARTING SEARCH: {prompt} ---")

#     # Download Video
#     video_path = "temp_video.mp4"
#     r = requests.get(video_url, stream=True)
#     with open(video_path, "wb") as f:
#         for chunk in r.iter_content(1024 * 1024):
#             f.write(chunk)

#     cap = cv2.VideoCapture(video_path)
#     results_list = []
#     frame_id = 0
    
#     # PERFORMANCE TUNING FOR CPU
#     # Skip more frames to prevent the server from hanging
#     skip_interval = 10 

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
        
#         frame_id += 1
#         if frame_id % skip_interval != 0: continue
        
#         print(f"Processing Frame: {frame_id}")

#         # 1. Detect using GroundingDINO
#         boxes = run_groundingdino(frame, prompt)

#         for box in boxes:
#             x1, y1, x2, y2 = box
#             crop = frame[y1:y2, x1:x2]
#             if crop.size == 0: continue

#             # 2. Verify with CLIP (Ensures 'RED' is actually 'RED')
#             score = verify_with_clip(crop, prompt)
            
#             if score > 0.23: # Threshold for CLIP matching
#                 img_name = f"match_f{frame_id}.jpg"
#                 img_path = os.path.join(SAVE_DIR, img_name)
                
#                 # Draw box on a copy for saving
#                 annotated = frame.copy()
#                 cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
#                 cv2.putText(annotated, f"{prompt} ({score:.2f})", (x1, y1-10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 cv2.imwrite(img_path, annotated)

#                 results_list.append({
#                     "object": prompt,
#                     "image_path": img_path,
#                     "bbox": [x1, y1, x2, y2],
#                     "timestamp": frame_id,
#                     "confidence": float(score)
#                 })

#     cap.release()
#     print(f"--- SEARCH FINISHED: Found {len(results_list)} matches ---")
#     return {"results": results_list}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import os
import warnings
import cv2
import torch
import requests
import numpy as np
from fastapi import FastAPI, Request
from PIL import Image

# GroundingDINO and Transforms
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
import open_clip

# 1. Sabhi faltu warnings ko disable karna
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DEVICE = "cpu" # Agar GPU hai toh "cuda" use karein
app = FastAPI()

# -----------------------------
# MODELS SETUP
# -----------------------------
print("AI Models Load ho rahe hain... Please wait.")

# GroundingDINO: Yeh aapke prompt (text) ko image se match karta hai
ground_model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py", 
    "groundingdino_swint_ogc.pth"
)
ground_model.to(DEVICE)

# CLIP: Color aur extra details verify karne ke liye
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model.to(DEVICE).eval()

SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# DETECTION ENGINE
# -----------------------------
def run_detection(frame, prompt):
    # Image ko Tensor mein badalna (Fixes the NumPy error)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_source = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_transformed, _ = transform(image_source, None)

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=ground_model,
            image=image_transformed,
            caption=prompt.lower(),
            box_threshold=0.35,
            text_threshold=0.25,
            device=DEVICE
        )
    
    h, w, _ = frame.shape
    pixel_boxes = []
    for i, box in enumerate(boxes):
        # Scale coordinates to frame size
        box = box * torch.Tensor([w, h, w, h])
        cx, cy, bw, bh = box.tolist()
        x1, y1 = int(cx - bw/2), int(cy - bh/2)
        x2, y2 = int(x1 + bw), int(y1 + bh)
        pixel_boxes.append({
            "bbox": [max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
            "conf": float(logits[i])
        })
        
    return pixel_boxes

def get_clip_score(crop, prompt):
    """CLIP se verify karna ki prompt match ho raha hai ya nahi"""
    image_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    image_input = clip_preprocess(image_pil).unsqueeze(0).to(DEVICE)
    text_input = clip_tokenizer([prompt]).to(DEVICE)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()
    return similarity

# -----------------------------
# MAIN API ENDPOINT
# -----------------------------
@app.post("/process")
async def process_video(req: Request):
    data = await req.json()
    video_url = data.get("fileUrl")
    prompt = data.get("prompt", "person").upper()
    
    print(f"Searching for: {prompt}")

    # 1. Video Download
    video_path = "temp_video.mp4"
    r = requests.get(video_url, stream=True)
    with open(video_path, "wb") as f:
        for chunk in r.iter_content(1024*1024): f.write(chunk)

    cap = cv2.VideoCapture(video_path)
    results_list = []
    frame_id = 0
    
    # PERFORMANCE: Har 12th frame check karega (CPU par speed ke liye)
    skip_frames = 12 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_id += 1
        if frame_id % skip_frames != 0: continue
        
        print(f"Processing frame {frame_id}...")

        # 2. Run GroundingDINO
        detections = run_detection(frame, prompt)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            # 3. CLIP Verification (Optional check for more accuracy)
            score = get_clip_score(crop, prompt)
            
            # Agar score 0.22 se zyada hai, matlab match mil gaya
            if score > 0.22:
                img_name = f"match_{frame_id}.jpg"
                img_path = os.path.join(SAVE_DIR, img_name)
                
                # Draw box and label
                annotated = frame.copy()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{prompt} {score:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(img_path, annotated)

                results_list.append({
                    "object": prompt,
                    "image_path": img_path,
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": frame_id,
                    "confidence": score
                })

    cap.release()
    print(f"Found {len(results_list)} matches.")
    return {"results": results_list}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)