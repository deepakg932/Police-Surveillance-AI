from fastapi import FastAPI, Request
from ultralytics import YOLO
import cv2
import os
import torch
import requests
import easyocr
import numpy as np
from difflib import SequenceMatcher
import open_clip
from groundingdino.util.inference import load_model, predict
from PIL import Image

app = FastAPI()

# -----------------------------
# OCR
# -----------------------------
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# -----------------------------
# YOLO MODELS
# -----------------------------
car_model = YOLO("yolov8n.pt")
plate_model = YOLO("license_plate_detector.pt")

# -----------------------------
# GroundingDINO MODEL
# -----------------------------
gdino_model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "groundingdino_swint_ogc.pth",
    device="cpu"
)

# -----------------------------
# CLIP MODEL
# -----------------------------
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

# -----------------------------
# SAVE DIRECTORY
# -----------------------------
SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    if not text:
        return ""
    return text.strip().upper()

# -----------------------------
# CHECK IF PROMPT IS PLATE SEARCH
# -----------------------------
def is_plate_query(prompt):

    for c in prompt:
        if c.isdigit():
            return True

    if len(prompt) <= 4:
        return True

    return False

# -----------------------------
# OCR FUNCTION
# -----------------------------
def perform_ocr(frame, box):

    x1,y1,x2,y2 = map(int,box)

    crop = frame[y1:y2,x1:x2]

    if crop.size == 0:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    results = reader.readtext(
        gray,
        detail=0,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    if len(results)==0:
        return ""

    return results[0].upper()

# -----------------------------
# STRING SIMILARITY
# -----------------------------
def similar(a,b):
    return SequenceMatcher(None,a,b).ratio()

# -----------------------------
# COLOR DETECTION
# -----------------------------
def detect_color(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "RED":[(0,120,70),(10,255,255)],
        "BLUE":[(94,80,2),(126,255,255)],
        "GREEN":[(25,52,72),(102,255,255)],
        "BLACK":[(0,0,0),(180,255,50)],
        "WHITE":[(0,0,200),(180,20,255)],
        "YELLOW":[(15,100,100),(35,255,255)]
    }

    for color,(lower,upper) in color_ranges.items():

        lower=np.array(lower)
        upper=np.array(upper)

        mask=cv2.inRange(hsv,lower,upper)

        ratio=cv2.countNonZero(mask)/(img.size/3)

        if ratio>0.1:
            return color

    return "UNKNOWN"

# -----------------------------
# DYNAMIC PROMPT DETECTION
# -----------------------------
def detect_prompt_objects(frame, prompt):

    # convert BGR → RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # convert numpy → torch tensor
    image = torch.from_numpy(image_rgb).permute(2,0,1).float()

    boxes, logits, phrases = predict(
        model=gdino_model,
        image=image,
        caption=prompt,
        box_threshold=0.3,
        text_threshold=0.25,
        device="cpu"
    )

    results = []

    for box in boxes:
        x1,y1,x2,y2 = map(int, box)

        if x2 <= x1 or y2 <= y1:
            continue

        results.append((x1,y1,x2,y2))

    return results

# -----------------------------
# MAIN API
# -----------------------------
@app.post("/process")
async def process_video(req: Request):

    data = await req.json()

    video_url = data.get("fileUrl")
    prompt = data.get("prompt","person")

    print("Searching:",prompt)

    video_path="temp_video.mp4"

    r=requests.get(video_url,stream=True)

    with open(video_path,"wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)

    cap=cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error":"video open failed"}

    results_list=[]
    frame_id=0
    saved=set()

    # =========================================
    # PLATE SEARCH MODE
    # =========================================
    if is_plate_query(prompt):

        print("Plate Search Mode")

        while cap.isOpened():

            ret,frame=cap.read()

            if not ret:
                break

            frame_id+=1

            if frame_id%2!=0:
                continue

            car_results=car_model(frame,classes=[2,3,5,7],conf=0.4)

            for car in car_results[0].boxes.xyxy:

                x1,y1,x2,y2=map(int,car)

                car_crop=frame[y1:y2,x1:x2]

                plate_results=plate_model(car_crop,conf=0.25)

                for plate in plate_results[0].boxes.xyxy:

                    px1,py1,px2,py2=map(int,plate)

                    px1+=x1
                    py1+=y1
                    px2+=x1
                    py2+=y1

                    ocr_text=perform_ocr(frame,[px1,py1,px2,py2])

                    if not ocr_text:
                        continue

                    if prompt in ocr_text or similar(prompt,ocr_text)>0.7:

                        if ocr_text in saved:
                            continue

                        saved.add(ocr_text)

                        img_path=os.path.join(
                            SAVE_DIR,
                            f"plate_{frame_id}.jpg"
                        )

                        annotated=frame.copy()

                        cv2.rectangle(
                            annotated,
                            (px1,py1),
                            (px2,py2),
                            (0,255,0),
                            2
                        )

                        cv2.putText(
                            annotated,
                            ocr_text,
                            (px1,py1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,0),
                            2
                        )

                        cv2.imwrite(img_path,annotated)

                        results_list.append({
                            "object":"license_plate",
                            "ocr_text":ocr_text,
                            "image_path":img_path,
                            "bbox":[px1,py1,px2,py2],
                            "timestamp":frame_id
                        })

    # =========================================
    # DYNAMIC PROMPT MODE
    # =========================================
    else:

        print("Dynamic Prompt Mode")

        while cap.isOpened():

            ret,frame=cap.read()

            if not ret:
                break

            frame_id+=1

            if frame_id%3!=0:
                continue

            boxes = detect_prompt_objects(frame, prompt)

            for box in boxes:

                x1,y1,x2,y2 = box

                crop=frame[y1:y2,x1:x2]

                color=detect_color(crop)

                img_path=os.path.join(
                    SAVE_DIR,
                    f"prompt_{frame_id}.jpg"
                )

                annotated=frame.copy()

                label=f"{prompt}"

                cv2.rectangle(
                    annotated,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

                cv2.imwrite(img_path,annotated)

                results_list.append({
                    "prompt":prompt,
                    "color":color,
                    "image_path":img_path,
                    "bbox":[x1,y1,x2,y2],
                    "timestamp":frame_id
                })

    cap.release()

    return {"results":results_list}