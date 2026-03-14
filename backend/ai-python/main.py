from fastapi import FastAPI, Request
from ultralytics import YOLO
import cv2
import os
import torch
import requests
import easyocr
import numpy as np
from difflib import SequenceMatcher
from PIL import Image
from groundingdino.util.inference import load_model, predict

app = FastAPI()

# -----------------------------
# OCR
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)

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
# SAVE DIRECTORY
# -----------------------------
SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------
# TEXT CLEAN
# -----------------------------
def clean_text(text):
    if not text:
        return ""
    return text.strip().upper()


# -----------------------------
# CHECK PLATE QUERY
# -----------------------------
def is_plate_query(prompt):

    for c in prompt:
        if c.isdigit():
            return True

    if len(prompt) <= 4:
        return True

    return False


# -----------------------------
# OCR
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
# SIMILARITY
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
# PROMPT DETECTION
# -----------------------------
def detect_prompt_objects(frame, prompt):

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = torch.from_numpy(image_rgb).permute(2,0,1).float()/255
    image = image.unsqueeze(0)

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

    # -----------------------------
    # LICENSE PLATE MODE
    # -----------------------------
    if is_plate_query(prompt):

        print("Plate Mode")

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

                        img_path=os.path.join(
                            SAVE_DIR,
                            f"plate_{frame_id}.jpg"
                        )

                        cv2.imwrite(img_path,frame)

                        results_list.append({
                            "object":"license_plate",
                            "ocr_text":ocr_text,
                            "image_path":img_path,
                            "timestamp":frame_id
                        })

    # -----------------------------
    # PROMPT MODE
    # -----------------------------
    else:

        print("Prompt Mode")

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

                cv2.imwrite(img_path,frame)

                results_list.append({
                    "prompt":prompt,
                    "color":color,
                    "image_path":img_path,
                    "bbox":[x1,y1,x2,y2],
                    "timestamp":frame_id
                })

    cap.release()

    return {"results":results_list}