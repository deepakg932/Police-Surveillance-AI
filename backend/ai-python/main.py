from fastapi import FastAPI, Request
import os
import cv2
import torch
import requests
import easyocr
import numpy as np
from difflib import SequenceMatcher
import open_clip
from groundingdino.util.inference import load_model, predict
from ultralytics import YOLO
from PIL import Image

# =========================
# FORCE CPU MODE
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = "cpu"

app = FastAPI()

# -----------------------------
# OCR
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)

# -----------------------------
# YOLO MODELS
# -----------------------------
car_model = YOLO("yolov8n.pt").to(DEVICE)
plate_model = YOLO("license_plate_detector.pt").to(DEVICE)

# -----------------------------
# GROUNDINGDINO
# -----------------------------
ground_model = load_model(
    "GroundingDINO_SwinT_OGC.py",
    "groundingdino_swint_ogc.pth"
)
ground_model.to(DEVICE)

# -----------------------------
# CLIP
# -----------------------------
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model.eval()
clip_model.to(DEVICE)

SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# UTILS
# -----------------------------
def clean_text(text):
    if not text:
        return ""
    return text.strip().upper()

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def is_plate_query(prompt):
    for c in prompt:
        if c.isdigit():
            return True
    if len(prompt) <= 4:
        return True
    return False

def detect_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "RED": [(0, 120, 70), (10, 255, 255)],
        "BLUE": [(94, 80, 2), (126, 255, 255)],
        "GREEN": [(25, 52, 72), (102, 255, 255)],
        "BLACK": [(0, 0, 0), (180, 255, 50)],
        "WHITE": [(0, 0, 200), (180, 20, 255)],
        "YELLOW": [(15, 100, 100), (35, 255, 255)]
    }
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        ratio = cv2.countNonZero(mask) / (img.size / 3)
        if ratio > 0.1:
            return color
    return "UNKNOWN"

def perform_ocr(frame, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(
        gray,
        detail=0,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    return results[0].upper() if results else ""

def clip_similarity(image, prompt):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_input = clip_preprocess(image).unsqueeze(0).to(DEVICE)
    text = clip_tokenizer([prompt])
    text = {k: v.to(DEVICE) for k, v in text.items()}
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()
    return similarity

def run_groundingdino(frame, prompt):
    boxes, logits, phrases = predict(
        model=ground_model,
        image=frame,
        caption=prompt,
        box_threshold=0.35,
        text_threshold=0.25,
        device=DEVICE
    )
    return boxes, logits, phrases

# -----------------------------
# MAIN API ENDPOINT
# -----------------------------
@app.post("/process")
async def process_video(req: Request):
    data = await req.json()
    video_url = data.get("fileUrl")
    prompt = clean_text(data.get("prompt"))
    print("Searching:", prompt)

    video_path = "temp_video.mp4"
    r = requests.get(video_url, stream=True)
    with open(video_path, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "video open failed"}

    results_list = []
    frame_id = 0
    saved = set()
    max_frames = 500

    # ======================================
    # NUMBER PLATE SEARCH
    # ======================================
    if is_plate_query(prompt):
        print("Plate Search Mode")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if frame_id % 5 != 0:
                continue
            if frame_id > max_frames:
                break
            car_results = car_model(frame, classes=[2,3,5,7], conf=0.4)
            for car in car_results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, car)
                car_crop = frame[y1:y2, x1:x2]
                plate_results = plate_model(car_crop, conf=0.25)
                for plate in plate_results[0].boxes.xyxy:
                    px1, py1, px2, py2 = map(int, plate)
                    px1 += x1
                    py1 += y1
                    px2 += x1
                    py2 += y1
                    ocr_text = perform_ocr(frame, [px1, py1, px2, py2])
                    if not ocr_text:
                        continue
                    if prompt in ocr_text or similar(prompt, ocr_text) > 0.75:
                        if ocr_text in saved:
                            continue
                        saved.add(ocr_text)
                        img_path = os.path.join(SAVE_DIR, f"plate_{frame_id}.jpg")
                        annotated = frame.copy()
                        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0,255,0),2)
                        cv2.putText(annotated, ocr_text, (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
                        cv2.imwrite(img_path, annotated)
                        results_list.append({
                            "object": "license_plate",
                            "ocr_text": ocr_text,
                            "image_path": img_path,
                            "bbox": [px1, py1, px2, py2],
                            "timestamp": frame_id
                        })

    # ======================================
    # DYNAMIC PROMPT DETECTION
    # ======================================
    else:
        print("Prompt Detection Mode")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if frame_id % 3 != 0:
                continue
            boxes, logits, phrases = run_groundingdino(frame, prompt)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                similarity = clip_similarity(crop, prompt)
                if similarity < 0.25:
                    continue
                img_path = os.path.join(SAVE_DIR, f"{prompt}_{frame_id}.jpg")
                annotated = frame.copy()
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0),2)
                cv2.putText(annotated, prompt, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                cv2.imwrite(img_path, annotated)
                results_list.append({
                    "object": prompt,
                    "image_path": img_path,
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": frame_id
                })

    cap.release()
    return {"results": results_list}