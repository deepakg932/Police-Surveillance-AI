
from fastapi import FastAPI, Request
from ultralytics import YOLO, YOLOWorld
import cv2
import os
import urllib.request
import torch
import numpy as np
import requests
import easyocr
import re
from pathlib import Path
from difflib import SequenceMatcher
from torchvision import models, transforms
from PIL import Image
from deepface import DeepFace



app = FastAPI()

# --------------------------------
# DEVICE
# --------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

# --------------------------------
# MODELS
# --------------------------------
car_model   = YOLO("yolov8n.pt").to(device)
plate_model = YOLO("license_plate_detector.pt").to(device)
world_model = YOLOWorld("yolov8s-worldv2.pt").to(device)
reader      = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Initialize face cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)



face_cascade_alt = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)
# --------------------------------
# FEATURE EXTRACTOR
# --------------------------------
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# --------------------------------
# CONFIG
# --------------------------------
SAVE_DIR       = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

FRAME_SKIP     = 1
AREA_THRESHOLD = 100
BLUR_THRESHOLD = 5
COOLDOWN       = 5
REID_THRESHOLD = 0.55   # ✅ INCREASED from 0.45 for better accuracy
REID_COOLDOWN  = 5     # frames between same-person saves

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================================
# COLOR THRESHOLDS
# ================================
COLOR_THRESHOLDS = {
    "white":  0.30,
    "silver": 0.25,
    "black":  0.25,
    "grey":   0.22,
    "gray":   0.22,
    "red":    0.12,
    "blue":   0.10,
    "green":  0.12,
    "yellow": 0.12,
    "orange": 0.12,
    "purple": 0.10,
    "pink":   0.10,
    "brown":  0.12,
    "maroon": 0.11,
    "navy":   0.10,
    "cyan":   0.09,
    "teal":   0.09,
    "olive":  0.10,
    "cream":  0.18,
    "beige":  0.16,
}

# ================================
# ALL INDIA STATE / UT CODES
# ================================
INDIA_STATE_CODES = {
    "AP":"Andhra Pradesh",  "AR":"Arunachal Pradesh",
    "AS":"Assam",           "BR":"Bihar",
    "CG":"Chhattisgarh",    "GA":"Goa",
    "GJ":"Gujarat",         "HR":"Haryana",
    "HP":"Himachal Pradesh","JH":"Jharkhand",
    "KA":"Karnataka",       "KL":"Kerala",
    "MP":"Madhya Pradesh",  "MH":"Maharashtra",
    "MN":"Manipur",         "ML":"Meghalaya",
    "MZ":"Mizoram",         "NL":"Nagaland",
    "OD":"Odisha",          "OR":"Odisha (old)",
    "PB":"Punjab",          "RJ":"Rajasthan",
    "SK":"Sikkim",          "TN":"Tamil Nadu",
    "TS":"Telangana",       "TR":"Tripura",
    "UP":"Uttar Pradesh",   "UK":"Uttarakhand",
    "WB":"West Bengal",     "AN":"Andaman & Nicobar",
    "CH":"Chandigarh",      "DN":"Dadra & Nagar Haveli",
    "DD":"Daman & Diu",     "DL":"Delhi",
    "JK":"Jammu & Kashmir", "LA":"Ladakh",
    "LD":"Lakshadweep",     "PY":"Puducherry",
}

OCR_SUBS = {
    '0':'O','O':'0','1':'I','I':'1','8':'B','B':'8',
    '5':'S','S':'5','2':'Z','Z':'2','6':'G','G':'6',
    '4':'A','D':'0',
}

PLATE_FULL_RE = re.compile(
    r'^([A-Z]{2})\s?(\d{1,2})\s?([A-Z]{1,3})\s?(\d{1,4})$'
)

def ensure_facenet_weights():
    """Download Facenet weights if missing"""
    weights_path = Path("/root/.deepface/weights/facenet_weights.h5")
    if not weights_path.exists():
        print("📥 Downloading Facenet weights...")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5"
        try:
            urllib.request.urlretrieve(url, weights_path)
            print("✅ Facenet weights downloaded successfully")
        except Exception as e:
            print(f"⚠️ Could not download Facenet weights: {e}")

# Call this at startup
ensure_facenet_weights()

# ================================
# PLATE UTILITIES
# ================================
def clean_plate(text):
    if not text:
        return ""
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def ocr_variants(text):
    variants = {text}
    for i, ch in enumerate(text):
        if ch in OCR_SUBS:
            alt = text[:i] + OCR_SUBS[ch] + text[i+1:]
            variants.add(alt)
    return variants

def parse_plate_string(raw):
    s = re.sub(r'\s+', '', raw)
    m = PLATE_FULL_RE.match(s)
    if not m:
        return None
    return {
        "state":  m.group(1),
        "rto":    m.group(2).zfill(2),
        "series": m.group(3),
        "number": m.group(4),
        "full":   m.group(1)+m.group(2).zfill(2)+m.group(3)+m.group(4)
    }

# ================================
# PROMPT TYPE DETECTION FOR PLATES
# ================================
def detect_plate_prompt_type(prompt: str):
    raw   = prompt.strip().upper()
    clean = re.sub(r'[^A-Z0-9]', '', raw)
    parts = raw.split()

    if clean in INDIA_STATE_CODES and len(clean) == 2:
        return {"type": "state_only", "state": clean,
                "desc": f"All vehicles from {INDIA_STATE_CODES[clean]}"}

    m = re.match(r'^([A-Z]{2})\s?(\d{1,2})$', raw)
    if m and m.group(1) in INDIA_STATE_CODES:
        st = m.group(1); rt = m.group(2).zfill(2)
        return {"type": "state_rto", "state": st, "rto": rt,
                "prefix": st+rt, "desc": f"RTO {st}-{rt}"}

    parsed = parse_plate_string(raw)
    if parsed and parsed["state"] in INDIA_STATE_CODES:
        return {"type": "full_plate", "parsed": parsed,
                "desc": f"Exact plate {parsed['full']}"}

    if len(parts) == 2:
        st_part  = re.sub(r'[^A-Z]', '', parts[0])
        num_part = re.sub(r'[^0-9]', '', parts[1])
        if st_part in INDIA_STATE_CODES and num_part:
            return {"type": "state_number", "state": st_part,
                    "number": num_part, "desc": f"State={st_part} ends {num_part}"}

    if any(c.isdigit() for c in clean) and clean not in INDIA_STATE_CODES:
        return {"type": "half_plate", "partial": clean,
                "desc": f"Plate contains '{clean}'"}

    return {"type": "not_plate"}

def is_plate_query(prompt: str) -> bool:
    return detect_plate_prompt_type(prompt)["type"] != "not_plate"

# ================================
# SMART PLATE MATCHING
# ================================
def smart_plate_match(plate_info: dict, ocr_plate: str) -> bool:
    plate = clean_plate(ocr_plate)
    if not plate or len(plate) < 2:
        return False

    ptype      = plate_info["type"]
    plate_vars = ocr_variants(plate)

    if ptype == "state_only":
        state = plate_info["state"]
        for v in plate_vars:
            if v.startswith(state): return True
        if len(plate) >= 2:
            if SequenceMatcher(None, state, plate[:2]).ratio() >= 0.80:
                return True
        return False

    elif ptype == "state_rto":
        prefix = plate_info["prefix"]
        for v in plate_vars:
            if v.startswith(prefix): return True
        if len(plate) >= 4:
            if SequenceMatcher(None, prefix, plate[:4]).ratio() >= 0.78:
                return True
        if len(plate) >= 4:
            sm = SequenceMatcher(None, plate_info["state"], plate[:2]).ratio() >= 0.80
            rm = SequenceMatcher(None, plate_info["rto"],   plate[2:4]).ratio() >= 0.80
            if sm and rm: return True
        return False

    elif ptype == "full_plate":
        target      = plate_info["parsed"]["full"]
        target_vars = ocr_variants(target)
        if plate == target: return True
        for tv in target_vars:
            if plate == tv: return True
        for tv in target_vars:
            if tv in plate or plate in tv: return True
        if SequenceMatcher(None, target, plate).ratio() >= 0.72:
            return True
        return False

    elif ptype == "state_number":
        state  = plate_info["state"]
        number = plate_info["number"]
        state_ok = False; num_ok = False
        for v in plate_vars:
            if v.startswith(state): state_ok = True
            if number in v:         num_ok   = True
        if not state_ok and len(plate) >= 2:
            state_ok = SequenceMatcher(None, state, plate[:2]).ratio() >= 0.80
        if not num_ok:
            tail   = plate[-len(number):]
            num_ok = SequenceMatcher(None, number, tail).ratio() >= 0.80
        return state_ok and num_ok

    elif ptype == "half_plate":
        partial      = plate_info["partial"]
        partial_vars = ocr_variants(partial)
        for pv in partial_vars:
            if pv in plate: return True
            for v in plate_vars:
                if pv in v: return True
        if SequenceMatcher(None, partial, plate).ratio() >= 0.75:
            return True
        if len(partial) <= 6 and len(plate) >= len(partial):
            for i in range(len(plate) - len(partial) + 1):
                window = plate[i:i+len(partial)]
                if SequenceMatcher(None, partial, window).ratio() >= 0.82:
                    return True
        return False

    return False

# ================================
# ENHANCED OCR FOR PLATES
# ================================
def enhance_plate_crop(crop):
    versions = []
    h, w = crop.shape[:2]
    scale = max(1.0, 200 / max(w, 1))
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, v1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(v1)
    versions.append(cv2.bitwise_not(v1))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    v3    = clahe.apply(gray)
    v3    = cv2.GaussianBlur(v3, (3, 3), 0)
    versions.append(v3)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    v4     = cv2.filter2D(gray, -1, kernel)
    versions.append(v4)
    return versions

def perform_ocr_plate(frame, box):
    x1, y1, x2, y2 = map(int, box)
    pad = 8
    x1  = max(0, x1-pad); y1 = max(0, y1-pad)
    x2  = min(frame.shape[1], x2+pad)
    y2  = min(frame.shape[0], y2+pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return ""
    enhanced_versions = enhance_plate_crop(crop)
    best_result = ""; best_score = 0
    for enhanced in enhanced_versions:
        results = reader.readtext(
            enhanced, detail=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        )
        if not results: continue
        combined = clean_plate("".join(results))
        if 5 <= len(combined) <= 12:
            score = len(combined)
            if len(combined) >= 2 and combined[:2] in INDIA_STATE_CODES:
                score += 5
            if score > best_score:
                best_score = score; best_result = combined
        for part in results:
            p = clean_plate(part)
            if len(p) >= 4:
                score = len(p)
                if len(p) >= 2 and p[:2] in INDIA_STATE_CODES:
                    score += 5
                if score > best_score:
                    best_score = score; best_result = p
    return best_result

# ================================
# COLOR RANGES (HSV)
# ================================
COLOR_RANGES = {
    "red":    [([0,   100, 100], [10,  255, 255]),
               ([160, 100, 100], [180, 255, 255])],
    "yellow": [([18,  100,  80], [35,  255, 255])],
    "blue":   [([95,   80,  50], [135, 255, 255])],
    "green":  [([35,   60,  40], [90,  255, 255])],
    "white":  [([0,     0, 180], [180,  40, 255])],
    "black":  [([0,     0,   0], [180, 255,  60])],
    "orange": [([8,   120, 100], [20,  255, 255])],
    "purple": [([125,  50,  50], [165, 255, 255])],
    "pink":   [([155,  30, 130], [175, 255, 255])],
    "silver": [([0,     0, 160], [180,  30, 255])],
    "grey":   [([0,     0,  60], [180,  25, 200])],
    "gray":   [([0,     0,  60], [180,  25, 200])],
    "brown":  [([8,    50,  20], [22,  200, 160])],
    # Common aliases/shades mapped to closest base ranges
    "maroon": [([0,   100, 100], [10,  255, 255]),
               ([160, 100, 100], [180, 255, 255])],
    "navy":   [([95,   80,  50], [135, 255, 255])],
    "cyan":   [([85,   60,  60], [100, 255, 255])],
    "teal":   [([85,   60,  60], [100, 255, 255])],
    "olive":  [([35,   60,  40], [90,  255, 255])],
    "cream":  [([18,   20, 180], [40,   90, 255])],
    "beige":  [([12,   20, 150], [30,  110, 255])],
}

COLOR_WORD_ALIASES = {
    "sky blue": "blue",
    "light blue": "blue",
    "dark blue": "blue",
    "navy blue": "navy",
    "dark green": "green",
    "light green": "green",
    "dark red": "red",
    "light red": "red",
    "dark purple": "purple",
    "light purple": "purple",
    "light pink": "pink",
    "dark pink": "pink",
    "off white": "white",
    "off-white": "white",
    "light gray": "gray",
    "light grey": "grey",
    "dark gray": "gray",
    "dark grey": "grey",
}

# ================================
# VEHICLE ALIASES
# ================================
VEHICLE_ALIASES = {
    "car":           {"strategy": "yolo_class", "class_ids": [2]},
    "truck":         {"strategy": "yolo_class", "class_ids": [7]},
    "bus":           {"strategy": "yolo_class", "class_ids": [5]},
    "van":           {"strategy": "yolo_class", "class_ids": [2, 7]},
    "bike":          {"strategy": "yolo_class", "class_ids": [3]},
    "bicycle":       {"strategy": "yolo_class", "class_ids": [1]},
    "cycle":         {"strategy": "yolo_class", "class_ids": [1]},
    "motorcycle":    {"strategy": "yolo_class", "class_ids": [3]},
    "motorbike":     {"strategy": "yolo_class", "class_ids": [3]},
    "scooter":       {"strategy": "yolo_class", "class_ids": [3]},
    "vehicle":       {"strategy": "yolo_class", "class_ids": [2, 3, 5, 7]},
    "auto":          {"strategy": "world", "world_label": "auto rickshaw"},
    "rickshaw":      {"strategy": "world", "world_label": "auto rickshaw"},
    "auto rickshaw": {"strategy": "world", "world_label": "auto rickshaw"},
    "tuk tuk":       {"strategy": "world", "world_label": "auto rickshaw"},
    "tuktuk":        {"strategy": "world", "world_label": "auto rickshaw"},
    "tempo":         {"strategy": "world", "world_label": "auto rickshaw"},
    "jeep":          {"strategy": "world", "world_label": "jeep"},
    "suv":           {"strategy": "world", "world_label": "SUV car"},
    "pickup":        {"strategy": "world", "world_label": "pickup truck"},
    "ambulance":     {"strategy": "world", "world_label": "ambulance"},
}

PERSON_ATTRIBUTE_KEYWORDS = [
    "helmet", "bag", "weapon", "gun", "knife", "cap", "hat",
    "backpack", "jacket", "mask", "uniform", "vest",
    "glasses", "sunglasses", "gloves", "shoe", "shoes", "handbag",
]

HEAD_ATTRS   = {"helmet", "cap", "hat", "mask", "glasses", "sunglasses"}
BODY_ATTRS   = {"bag", "backpack", "jacket", "vest", "uniform", "gloves", "shoes", "handbag"}
WEAPON_ATTRS = {"weapon", "gun", "knife"}

# Clothes keywords — "person wearing green SHIRT" type prompts ke liye
CLOTHES_KEYWORDS = {
    "shirt", "tshirt", "t-shirt", "top", "blouse", "kurta",
    "jacket", "coat", "hoodie", "sweater", "dress", "saree",
    "sari", "suit", "uniform", "clothes", "clothing", "outfit",
    "pants", "jeans", "trouser", "skirt", "shorts", "leggings",
}

# ================================
# GENERAL UTILITIES
# ================================
def extract_features(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        return resnet(img).flatten().numpy()


def compute_reid_similarity(feat1, feat2):
    """Cosine similarity between two feature vectors"""
    if feat1 is None or feat2 is None:
        return 0.0
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(feat1, feat2) / (norm1 * norm2))

def extract_person_features(frame, box):
    """✅ FIXED: Better person feature extraction with more padding"""
    x1, y1, x2, y2 = map(int, box)
    
    # More padding for better context
    pad = 20  # 10 से बढ़ाकर 20
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    
    # Try multiple resizing strategies
    try:
        # Strategy 1: Normal size
        feat1 = extract_features(crop)
        
        # Strategy 2: If crop is small, resize up
        h_c, w_c = crop.shape[:2]
        if h_c < 100 or w_c < 100:
            scale = max(1.0, 200 / max(w_c, 1))
            crop_resized = cv2.resize(crop, None, fx=scale, fy=scale)
            feat2 = extract_features(crop_resized)
            # Return average of both features
            return (feat1 + feat2) / 2
        
        return feat1
    except:
        return None


def is_blurry(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

def draw_box(frame, x1, y1, x2, y2, label, color=(0, 255, 0)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

def save_detection(frame, box, label, frame_id,
                   prefix="det", color=(0, 255, 0)):
    x1, y1, x2, y2 = box
    ann  = frame.copy()
    draw_box(ann, x1, y1, x2, y2, label, color)
    path = os.path.join(SAVE_DIR, f"{prefix}_{frame_id}.jpg")
    cv2.imwrite(path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def color_ratio(crop, color_name):
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES.get(color_name.lower(), []):
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv, np.array(lo, dtype=np.uint8),
                              np.array(hi, dtype=np.uint8))
        )
    return cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1] + 1e-5)

def detect_color_in_crop(crop, color_name, threshold=0.15):
    if crop is None or crop.size == 0:
        return False

    requested = str(color_name).lower().strip()
    ratio = color_ratio(crop, requested)
    if ratio >= threshold:
        return True

    # Small/blurred regions (helmet, bag, fast motion) often under-shoot threshold.
    # Accept if requested color is dominant with a slightly relaxed gate.
    dom = get_dominant_color_name(crop)
    relaxed_threshold = threshold * 0.70
    if dom == requested and ratio >= relaxed_threshold:
        return True

    return False

def get_dominant_color_name(crop):
    best, best_r = "unknown", 0.0
    for cname in COLOR_RANGES:
        r = color_ratio(crop, cname)
        if r > best_r:
            best, best_r = cname, r
    return best

def get_shoe_color_threshold(color_name):
    """Shoes are tiny in frame; keep slightly softer color thresholds."""
    c = str(color_name).lower().strip()
    if c in {"black", "white", "silver", "grey", "gray"}:
        return 0.10
    if c in {"orange", "red", "brown", "purple", "pink"}:
        return 0.09
    return 0.10

def extract_color_from_words(words_list):
    """List of words mein se pehla color word dhundho"""
    for w in words_list:
        if w.strip() in COLOR_RANGES:
            return w.strip()
    return None

def normalize_prompt_color_words(text):
    s = str(text).lower()
    # Replace longer phrases first.
    for src in sorted(COLOR_WORD_ALIASES.keys(), key=len, reverse=True):
        dst = COLOR_WORD_ALIASES[src]
        s = re.sub(rf"\b{re.escape(src)}\b", dst, s)
    return s

# ================================
# COOLDOWN TRACKER
# ================================
class CooldownTracker:
    def __init__(self, cooldown=COOLDOWN):
        self._t = {}
        self._c = cooldown

    def should_skip(self, key, fid):
        return key in self._t and fid - self._t[key] < self._c

    def update(self, key, fid):
        self._t[key] = fid

# ================================
# GENDER HELPER
# ================================
def predict_gender_deepface(crop):
    try:
        result = DeepFace.analyze(
            crop, actions=['gender'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        gender = result.get("dominant_gender", "").lower()
        if gender in ["man", "male"]:     return "male"
        if gender in ["woman", "female"]: return "female"
        return "unknown"
    except:
        return "unknown"

# ================================
# PROMPT ROUTER
# ================================
def find_vehicle_in_prompt(p):
    for alias in sorted(VEHICLE_ALIASES.keys(), key=len, reverse=True):
        if alias in p:
            return alias, VEHICLE_ALIASES[alias]
    return None, None

def parse_prompt(prompt: str):
    p = prompt.strip().lower()
    p = re.sub(r'\bauto\s*rickshaw\b', 'auto rickshaw', p)
    p = re.sub(r'\btuk[\s-]?tuk\b',   'tuk tuk', p)
    p = re.sub(r'\bshoe\b', 'shoes', p)
    p = re.sub(r'\bwear\b', 'wearing', p)
    p = re.sub(r'\bwears\b', 'wearing', p)
    p = normalize_prompt_color_words(p)

    # 0. ✅ IMAGE-BASED REID — if prompt says "find this person" / "match person" / "person in image"
    reid_triggers = [
        "find this person", "match person", "person in image",
        "same person", "this person", "find person",
        "identify person", "track this person", "search person"
    ]
    if any(trigger in p for trigger in reid_triggers):
        return {"mode": "person_reid", "prompt": prompt}

    # 1. PLATE CHECK
    plate_info = detect_plate_prompt_type(prompt.strip())
    if plate_info["type"] != "not_plate":
        return {"mode": "plate", "plate_info": plate_info, "prompt": prompt}

    detected_color   = next((c for c in COLOR_RANGES if c in p), None)
    any_color        = bool(re.search(r'\bany[\s_]?colou?r\b', p))
    veh_word, v_info = find_vehicle_in_prompt(p)

    # 2. Person WITHOUT attribute
    m = re.search(r'person\s+without\s+(\w+)', p)
    if m:
        attr = m.group(1).strip()
        if attr in PERSON_ATTRIBUTE_KEYWORDS or attr == "helmet":
            return {"mode": "person_without_attribute",
                    "attribute": attr, "prompt": prompt}

    # 3. Person with / wearing / on vehicle + COLOR helmet (helmet crop pe color)
    helmet_color = None
    m = re.search(r'person\s+with\s+(\w+)\s+helmet', p)
    if m:
        helmet_color = m.group(1).strip()
    if not helmet_color:
        m = re.search(
            r'person\s+(?:wearing|in)\s+(?:a|an|the)?\s*(\w+)\s+helmet', p)
        if m:
            helmet_color = m.group(1).strip()
    if not helmet_color:
        # e.g. "person on bike red helmet" — word before helmet = color
        m = re.search(
            r'person\s+(?:on|with|riding)\s+(.+)\s+(\w+)\s+helmet',
            p,
        )
        if m:
            helmet_color = m.group(2).strip()
    if helmet_color and helmet_color in COLOR_RANGES:
        return {"mode": "person_helmet_color",
                "color": helmet_color, "prompt": prompt}

    # 4. Person with COLOR bag/backpack/handbag
    m = re.search(
        r'person\s+(?:with|having|carrying|hold|holding)\s+(\w+)\s+(bag|backpack|handbag)', p)
    if m:
        color    = m.group(1).strip()
        bag_type = m.group(2).strip()
        if color in COLOR_RANGES:
            return {"mode": "person_bag_color",
                    "color": color, "bag_type": bag_type, "prompt": prompt}

    # 5. Text/poster search
    m = re.search(r'(text|poster|word)\s+(.+)', p)
    if m:
        return {"mode": "text_search",
                "target_text": m.group(2).strip().upper(), "prompt": prompt}

    # 6. Gender + Color
    m = re.search(r'(male|female)\s+(?:wear(?:ing)?|in)\s+(\w+)', p)
    if m:
        gender = m.group(1).strip()
        color  = m.group(2).strip()
        if color in COLOR_RANGES:
            return {"mode": "gender_color",
                    "gender": gender, "color": color, "prompt": prompt}

    # 6.1 Person wearing COLOR ATTRIBUTE
    # handles: "person wearing orange shoes", "person wearing a red jacket"
    m = re.search(
        r'person\s+(?:wearing|in)\s+(?:a|an|the)?\s*(\w+)\s+'
        r'(helmet|bag|backpack|handbag|jacket|mask|uniform|vest|glasses|sunglasses|gloves|shoes|shoe)',
        p
    )
    if m:
        color = m.group(1).strip()
        attr = m.group(2).strip()
        if attr == "shoe":
            attr = "shoes"
        if color in COLOR_RANGES:
            return {"mode": "person_attribute_color",
                    "attribute": attr, "color": color, "prompt": prompt}

    # 7. Person + Attribute + Color — e.g. "person with helmet wearing red"
    if "person" in p and "with" in p and ("wearing" in p or " in " in p):
        m = re.search(
            r'person\s+with\s+(\w+)\s+(?:wear(?:ing)?|in)\s+(\w+)', p)
        if m:
            attr  = m.group(1).strip()
            color = m.group(2).strip()
            if attr in PERSON_ATTRIBUTE_KEYWORDS and color in COLOR_RANGES:
                return {"mode": "person_attribute_color",
                        "attribute": attr, "color": color, "prompt": prompt}

    # 8. Person + WITH/ON + Vehicle (no attribute) — MUST come before attribute-only check
    if "person" in p and ("with" in p or "on" in p or "riding" in p) and veh_word:
        attr_found = any(
            re.search(rf'\b{re.escape(attr)}\b', p)
            for attr in PERSON_ATTRIBUTE_KEYWORDS
        )
        if not attr_found:
            return {"mode": "person_vehicle",
                    "vehicle_word": veh_word, "vehicle_info": v_info,
                    "prompt": prompt}

    # 8.5 Person + Attribute (no color) — e.g. "person with helmet"
    if "person" in p and "with" in p:
        for attr in sorted(PERSON_ATTRIBUTE_KEYWORDS, key=len, reverse=True):
            if re.search(rf'\b{re.escape(attr)}\b', p):
                return {"mode": "person_attribute_color",
                        "attribute": attr, "color": None, "prompt": prompt}


    # 9. Person + Vehicle + Color — e.g. "person on bike wearing red"
    if "person" in p and veh_word and ("wearing" in p or " in " in p):
        m = re.search(
            r'person\s+(?:on|with|riding)\s+' + re.escape(veh_word) +
            r'\s+(?:wear(?:ing)?|in)\s+(\w+)', p)
        if m:
            color = m.group(1).strip()
            if color in COLOR_RANGES:
                return {"mode": "person_vehicle_color",
                        "vehicle_word": veh_word, "vehicle_info": v_info,
                        "color": color, "prompt": prompt}

    # 10. ✅ Person wearing COLOR (any clothes keyword supported)
    # handles: "person wearing red", "person wearing green shirt",
    #          "person in blue", "person in red jacket" etc.
    m = re.search(r'person\s+(?:wear(?:ing)?|in)\s+(.+)', p)
    if m:
        rest  = m.group(1).strip()          # e.g. "green shirt" or "red"
        words = rest.split()
        color = extract_color_from_words(words)
        if color:
            return {"mode": "color_attribute",
                    "object": "person", "color": color, "prompt": prompt}

    # 11. Color + Attribute — e.g. "red helmet", "blue jacket"
    if detected_color:
        for attr in sorted(PERSON_ATTRIBUTE_KEYWORDS, key=len, reverse=True):
            if attr in p:
                return {"mode": "person_attribute_color",
                        "attribute": attr, "color": detected_color,
                        "prompt": prompt}

    # 12. Color + Vehicle — e.g. "red car", "blue bike"
    if (detected_color or any_color) and veh_word:
        return {"mode": "color_object", "color": detected_color,
                "vehicle_word": veh_word, "vehicle_info": v_info,
                "prompt": prompt}

    # 13. Person + attribute/vehicle
    if "person" in p:
        found = [kw for kw in PERSON_ATTRIBUTE_KEYWORDS if kw in p]
        if found:
            return {"mode": "person_attribute_color",
                    "attribute": found[0], "color": None, "prompt": prompt}
        if veh_word:
            return {"mode": "person_vehicle",
                    "vehicle_word": veh_word, "vehicle_info": v_info,
                    "prompt": prompt}

    # 14. Akela vehicle
    if veh_word:
        return {"mode": "color_object", "color": None,
                "vehicle_word": veh_word, "vehicle_info": v_info,
                "prompt": prompt}

    # 15. Fallback YOLOWorld
    return {"mode": "yoloworld", "prompt": prompt}

def _norm_words(text):
    return re.sub(r'[^a-z0-9\s]', ' ', str(text).lower()).split()

def _contains_word(text, word):
    return re.search(rf"\b{re.escape(word.lower())}\b", str(text).lower()) is not None

def _concat_detection_text(det):
    parts = [
        det.get("object", ""),
        det.get("attribute", ""),
        det.get("vehicle", ""),
        det.get("color", ""),
        det.get("ocr_text", ""),
        det.get("plate", ""),
    ]
    return " ".join(str(x) for x in parts if x).lower()

def _matches_parsed_prompt(det, parsed, raw_prompt):
    mode = parsed.get("mode", "")
    blob = _concat_detection_text(det)

    if mode == "plate":
        return ("license_plate" in blob) or bool(det.get("plate"))

    if mode == "text_search":
        target = str(parsed.get("target_text", "")).lower()
        return target and target in blob

    if mode == "person_without_attribute":
        attr = str(parsed.get("attribute", "")).lower()
        return ("person without" in blob) and (attr in blob)

    if mode == "person_vehicle":
        vehicle_word = str(parsed.get("vehicle_word", "")).lower().strip()
        return _contains_word(blob, vehicle_word) or _contains_word(blob, "person")

    if mode == "person_reid":
        return "matched person" in blob or det.get("similarity", 0) > 0

    # Common strict checks for modes with fields.
    color = str(parsed.get("color", "")).lower().strip()
    attr = str(parsed.get("attribute", "")).lower().strip()
    vehicle_word = str(parsed.get("vehicle_word", "")).lower().strip()
    gender = str(parsed.get("gender", "")).lower().strip()

    if attr and attr == "shoe":
        attr = "shoes"

    if attr and not (_contains_word(blob, attr) or (attr == "shoes" and _contains_word(blob, "shoe"))):
        return False
    if color and color != "any" and not _contains_word(blob, color):
        return False
    if vehicle_word and not _contains_word(blob, vehicle_word):
        return False
    if gender and not _contains_word(blob, gender):
        return False

    if mode == "yoloworld":
        stop = {"person", "with", "wearing", "in", "on", "having", "carrying", "a", "an", "the"}
        words = [w for w in _norm_words(raw_prompt) if len(w) > 2 and w not in stop]
        if not words:
            return True
        return any(_contains_word(blob, w) for w in words)

    return True

def _compute_color_histogram(img_bgr):
    """HSV color histogram for person appearance matching"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # H: 50 bins, S: 60 bins — ignore V (brightness/lighting invariant)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60],
                        [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def filter_results_by_prompt(results_list, parsed, raw_prompt):
    kept = []
    removed = []
    for det in results_list:
        if _matches_parsed_prompt(det, parsed, raw_prompt):
            kept.append(det)
        else:
            removed.append(det)

    # Faltu detections ki saved images bhi delete kar do.
    for det in removed:
        p = det.get("image_path")
        if not p:
            continue
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception as e:
            print(f"⚠️ Could not delete unmatched image {p}: {e}")

    return kept, removed

# ================================
# VEHICLE DETECTION HELPER
# ================================
def detect_vehicles_in_frame(frame, vehicle_info,
                              conf_yolo=0.35,
                              conf_world=0.30):
    boxes = []
    if vehicle_info["strategy"] == "yolo_class":
        res = car_model(frame, classes=vehicle_info["class_ids"],
                        conf=conf_yolo)
        if res[0].boxes is not None:
            for b in res[0].boxes.xyxy.cpu().numpy():
                boxes.append(list(map(int, b)))
    else:
        world_model.set_classes([vehicle_info["world_label"]])
        res = world_model(frame, conf=conf_world, imgsz=640)
        if res[0].boxes is not None:
            for b in res[0].boxes.xyxy.cpu().numpy():
                boxes.append(list(map(int, b)))
    return boxes

# ================================
# SPATIAL CHECKS
# ================================
def iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/(ua+1e-6)

def helmet_belongs_to_person(p_box, h_box):
    px1,py1,px2,py2=p_box; hx1,hy1,hx2,hy2=h_box
    hcx=(hx1+hx2)/2; hcy=(hy1+hy2)/2
    ph=py2-py1; pw=px2-px1
    hw = hx2 - hx1
    hh = hy2 - hy1
    if ph <= 0 or pw <= 0 or hw <= 0 or hh <= 0:
        return False

    # Rider videos may have motion blur/occlusion; keep spatial checks robust.
    in_head_x = (px1 - pw * 0.12) <= hcx <= (px2 + pw * 0.12)
    in_head_y = (py1 - ph * 0.18) <= hcy <= (py1 + ph * 0.48)
    size_ok   = (hw < pw * 0.85) and (hh < ph * 0.45)
    return in_head_x and in_head_y and size_ok

def bag_belongs_to_person(p_box, b_box):
    px1,py1,px2,py2=p_box
    bx1,by1,bx2,by2=b_box

    cx=(b_box[0]+b_box[2])/2
    cy=(b_box[1]+b_box[3])/2

    # ✅ Relaxed: Person bbox se 30% bahar allow karo
    margin_x = (px2-px1) * 0.30
    margin_y = (py2-py1) * 0.30

    return (px1-margin_x<=cx<=px2+margin_x) and (py1-margin_y<=cy<=py2+margin_y)

def weapon_belongs_to_person(p_box, w_box):
    px1,py1,px2,py2=p_box
    ew=(px2-px1)*0.3; eh=(py2-py1)*0.3
    cx=(w_box[0]+w_box[2])/2; cy=(w_box[1]+w_box[3])/2
    return (px1-ew<=cx<=px2+ew) and (py1-eh<=cy<=py2+eh)

def shoes_belongs_to_person(p_box, s_box):
    px1, py1, px2, py2 = p_box
    sx1, sy1, sx2, sy2 = s_box
    cx = (sx1 + sx2) / 2
    cy = (sy1 + sy2) / 2
    ph = py2 - py1
    pw = px2 - px1
    sw = sx2 - sx1
    sh = sy2 - sy1
    if ph <= 0 or pw <= 0:
        return False
    if sw <= 0 or sh <= 0:
        return False

    # Shoes should be near foot region and relatively small vs person bbox.
    if not (py1 + ph * 0.62 <= cy <= py2 + ph * 0.03):
        return False
    if not (px1 - pw * 0.06 <= cx <= px2 + pw * 0.06):
        return False
    if sw > pw * 0.65 or sh > ph * 0.30:
        return False
    # Avoid tiny noisy detections.
    if (sw * sh) < max(120, (pw * ph) * 0.002):
        return False
    return True

def attr_belongs_to_person(p_box, a_box, attr_name):
    if attr_name in HEAD_ATTRS:     return helmet_belongs_to_person(p_box, a_box)
    elif attr_name in WEAPON_ATTRS: return weapon_belongs_to_person(p_box, a_box)
    elif attr_name in {"shoe", "shoes"}: return shoes_belongs_to_person(p_box, a_box)
    else:                           return bag_belongs_to_person(p_box, a_box)

# ================================
# DETECTION MODES
# ================================

# ---- 1. LICENSE PLATE ----
def run_plate_mode(cap, plate_info, results_list):
    tracker  = CooldownTracker(cooldown=25)
    frame_id = 0
    ptype    = plate_info["type"]
    print(f"  Plate type: {ptype} | {plate_info.get('desc','')}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        found_plates = []

        direct_results = plate_model(frame, conf=0.15, imgsz=1280)
        if direct_results[0].boxes is not None:
            for pbox in direct_results[0].boxes.xyxy.cpu().numpy():
                px1,py1,px2,py2 = map(int, pbox)
                ocr = perform_ocr_plate(frame, [px1,py1,px2,py2])
                if ocr:
                    found_plates.append(([px1,py1,px2,py2], ocr, None))

        car_results = car_model(frame, classes=[2,3,5,7], conf=0.30)
        if car_results[0].boxes is not None:
            for cbox in car_results[0].boxes.xyxy.cpu().numpy():
                cx1,cy1,cx2,cy2 = map(int, cbox)
                car_crop = frame[cy1:cy2, cx1:cx2]
                if car_crop.size == 0: continue
                plate_in_car = plate_model(car_crop, conf=0.15)
                if plate_in_car[0].boxes is None: continue
                for pbox in plate_in_car[0].boxes.xyxy.cpu().numpy():
                    px1,py1,px2,py2 = map(int, pbox)
                    px1+=cx1; py1+=cy1; px2+=cx1; py2+=cy1
                    ocr = perform_ocr_plate(frame, [px1,py1,px2,py2])
                    if ocr:
                        found_plates.append(([px1,py1,px2,py2], ocr,
                                             [cx1,cy1,cx2,cy2]))

        h, w = frame.shape[:2]
        if w <= 1280:
            upscaled   = cv2.resize(frame, (w*2, h*2),
                                    interpolation=cv2.INTER_LINEAR)
            up_results = plate_model(upscaled, conf=0.20, imgsz=1280)
            if up_results[0].boxes is not None:
                for pbox in up_results[0].boxes.xyxy.cpu().numpy():
                    px1,py1,px2,py2 = map(int, pbox)
                    px1//=2; py1//=2; px2//=2; py2//=2
                    ocr = perform_ocr_plate(frame, [px1,py1,px2,py2])
                    if ocr:
                        found_plates.append(([px1,py1,px2,py2], ocr, None))

        for plate_box, ocr_text, car_box in found_plates:
            if not smart_plate_match(plate_info, ocr_text): continue
            key = f"plate_{plate_box[0]//60}_{plate_box[1]//60}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            state_name = ""
            if len(ocr_text) >= 2 and ocr_text[:2] in INDIA_STATE_CODES:
                state_name = INDIA_STATE_CODES[ocr_text[:2]]
            label = ocr_text + (f" ({state_name})" if state_name else "")

            ann = frame.copy()
            if car_box: draw_box(ann, *car_box, "vehicle", (255,200,0))
            draw_box(ann, *plate_box, label, (0,255,0))
            img_path = os.path.join(SAVE_DIR, f"plate_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"  ✅ Plate: {ocr_text} frame={frame_id}")

            results_list.append({
                "object":       "license_plate",
                "plate":        ocr_text,
                "bbox":         plate_box,
                "state":        ocr_text[:2] if len(ocr_text) >= 2 else "",
                "state_name":   state_name,
                "match_type":   ptype,
                "image_path":   img_path,
                "plate_bbox":   plate_box,
                "vehicle_bbox": car_box,
                "timestamp":    frame_id,
                "confidence":   0.90
            })


# ---- 0. PERSON RE-IDENTIFICATION (image-based) ✅ FIXED ----
# ---- 0. PERSON RE-IDENTIFICATION (UNIVERSAL FACE MATCHING) ----
def run_person_reid_mode(cap, ref_feat, results_list, ref_img_bgr=None, threshold=None):
    """✅ Universal Face ReID - Matches ANY face from ANY image"""
    
    if ref_img_bgr is None:
        print("  ❌ No reference image provided")
        return
    
    # ================================
    # EXTRACT REFERENCE FACE EMBEDDING
    # ================================
    ref_face_embed = None
    ref_face_detected = False
    ref_face_bbox = None
    
    print("  🔍 Extracting face from reference image...")
    
    # Try multiple face detection methods
    try:
        # Method 1: Haar Cascade (fast)
        gray_ref = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2GRAY)
        faces_ref = face_cascade.detectMultiScale(gray_ref, 1.1, 4)
        
        if len(faces_ref) > 0:
            # Get largest face
            fx, fy, fw, fh = max(faces_ref, key=lambda x: x[2]*x[3])
            face_roi = ref_img_bgr[fy:fy+fh, fx:fx+fw]
            ref_face_bbox = [fx, fy, fx+fw, fy+fh]
            
            # Get face embedding
            face_result = DeepFace.represent(
                face_roi,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="skip"
            )
            if face_result:
                ref_face_embed = np.array(face_result[0]["embedding"])
                ref_face_detected = True
                print(f"  ✅ Face detected! Size: {fw}x{fh}")
        
        # Method 2: If Haar fails, try DeepFace directly
        if not ref_face_detected:
            face_result = DeepFace.represent(
                ref_img_bgr,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="opencv"
            )
            if face_result:
                ref_face_embed = np.array(face_result[0]["embedding"])
                ref_face_detected = True
                print(f"  ✅ Face detected via DeepFace!")
                
    except Exception as e:
        print(f"  ⚠️ Face extraction failed: {e}")
    
    if not ref_face_detected:
        print("  ❌ No face found in reference image!")
        print("  💡 Tip: Use an image where face is clearly visible")
        return
    
    # Face matching threshold (adjustable)
    FACE_THRESHOLD = threshold if threshold else 0.55  # Default 55% similarity
    
    tracker = CooldownTracker(cooldown=REID_COOLDOWN)
    frame_id = 0
    matched_count = 0
    matched_faces = {}  # Track matched faces to avoid duplicates
    
    print(f"  🎯 Face matching threshold: {FACE_THRESHOLD}")
    print(f"  🔍 Searching video for matching face...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: 
            continue
        
        # Detect all persons first (for bounding box)
        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.25, imgsz=640)
        
        if not res_p[0].boxes:
            # If no person detected, still try direct face detection
            try:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_frame = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
                
                for (fx, fy, fw, fh) in faces_frame:
                    if fw * fh < 500:  # Skip tiny faces
                        continue
                        
                    face_roi = frame[fy:fy+fh, fx:fx+fw]
                    if face_roi.size == 0:
                        continue
                    
                    # Get face embedding
                    face_result = DeepFace.represent(
                        face_roi,
                        model_name="Facenet",
                        enforce_detection=False,
                        detector_backend="skip"
                    )
                    
                    if not face_result:
                        continue
                    
                    current_face_embed = np.array(face_result[0]["embedding"])
                    
                    # Calculate similarity
                    norm1 = np.linalg.norm(ref_face_embed)
                    norm2 = np.linalg.norm(current_face_embed)
                    if norm1 == 0 or norm2 == 0:
                        continue
                    
                    face_similarity = float(np.dot(ref_face_embed, current_face_embed) / (norm1 * norm2))
                    
                    if face_similarity >= FACE_THRESHOLD:
                        # Create unique key for this face position
                        face_key = f"face_{fx//30}_{fy//30}"
                        
                        if tracker.should_skip(face_key, frame_id):
                            continue
                        
                        tracker.update(face_key, frame_id)
                        matched_count += 1
                        
                        # Draw box around face
                        ann = frame.copy()
                        cv2.rectangle(ann, (fx, fy), (fx+fw, fy+fh), (0, 255, 255), 3)
                        label = f"MATCHED! ({face_similarity:.2f})"
                        cv2.putText(ann, label, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        img_path = os.path.join(SAVE_DIR, f"matched_face_{frame_id}_{matched_count}.jpg")
                        cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        print(f"  ✅ MATCH FOUND! Frame {frame_id} | Similarity: {face_similarity:.3f}")
                        
                        results_list.append({
                            "object": "matched_face",
                            "similarity": round(face_similarity, 3),
                            "confidence": round(face_similarity, 3),
                            "face_score": round(face_similarity, 3),
                            "bbox": [fx, fy, fx+fw, fy+fh],
                            "image_path": img_path,
                            "timestamp": frame_id,
                            "detection_method": "direct_face"
                        })
                        
            except Exception as e:
                continue
                
            continue  # Skip to next frame
        
        # Persons detected - check each person for face
        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
        
        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            area = (px2-px1)*(py2-py1)
            if area < 150:
                continue
            
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue
            
            # ================================
            # FACE DETECTION IN CURRENT PERSON
            # ================================
            try:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                faces_crop = face_cascade.detectMultiScale(gray_crop, 1.1, 4)
                
                if len(faces_crop) == 0:
                    # Try with different scale factor for small faces
                    faces_crop = face_cascade.detectMultiScale(gray_crop, 1.05, 3)
                
                if len(faces_crop) == 0:
                    continue
                
                # Process each face in this person
                best_face_score = 0
                best_face_bbox = None
                
                for (fx, fy, fw, fh) in faces_crop:
                    if fw * fh < 300:  # Skip very tiny faces
                        continue
                    
                    # Adjust face coordinates to original frame
                    global_fx = px1 + fx
                    global_fy = py1 + fy
                    global_fw = fw
                    global_fh = fh
                    
                    face_roi = crop[fy:fy+fh, fx:fx+fw]
                    if face_roi.size == 0:
                        continue
                    
                    # Get face embedding
                    face_result = DeepFace.represent(
                        face_roi,
                        model_name="Facenet",
                        enforce_detection=False,
                        detector_backend="skip"
                    )
                    
                    if not face_result:
                        continue
                    
                    current_face_embed = np.array(face_result[0]["embedding"])
                    
                    # Calculate face similarity
                    norm1 = np.linalg.norm(ref_face_embed)
                    norm2 = np.linalg.norm(current_face_embed)
                    
                    if norm1 == 0 or norm2 == 0:
                        continue
                    
                    face_similarity = float(np.dot(ref_face_embed, current_face_embed) / (norm1 * norm2))
                    
                    if face_similarity > best_face_score:
                        best_face_score = face_similarity
                        best_face_bbox = [global_fx, global_fy, global_fx+global_fw, global_fy+global_fh]
                
                # Check if best face match meets threshold
                if best_face_score >= FACE_THRESHOLD:
                    # Create unique key
                    face_key = f"face_{best_face_bbox[0]//30}_{best_face_bbox[1]//30}"
                    
                    if tracker.should_skip(face_key, frame_id):
                        continue
                    
                    tracker.update(face_key, frame_id)
                    matched_count += 1
                    
                    # Draw on frame
                    ann = frame.copy()
                    # Draw person box
                    draw_box(ann, px1, py1, px2, py2, "person", (0, 255, 0))
                    # Draw face box in yellow
                    cv2.rectangle(ann, (best_face_bbox[0], best_face_bbox[1]), 
                                  (best_face_bbox[2], best_face_bbox[3]), (0, 255, 255), 3)
                    
                    label = f"MATCHED! Face: {best_face_score:.2f}"
                    cv2.putText(ann, label, (best_face_bbox[0], best_face_bbox[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    img_path = os.path.join(SAVE_DIR, f"matched_{frame_id}_{matched_count}.jpg")
                    cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    print(f"  ✅ MATCH FOUND! Frame {frame_id} | Face similarity: {best_face_score:.3f}")
                    
                    results_list.append({
                        "object": "matched_person",
                        "similarity": round(best_face_score, 3),
                        "confidence": round(best_face_score, 3),
                        "face_score": round(best_face_score, 3),
                        "bbox": [px1, py1, px2, py2],
                        "face_bbox": best_face_bbox,
                        "image_path": img_path,
                        "timestamp": frame_id,
                        "detection_method": "person_face"
                    })
                    
            except Exception as e:
                # If face detection fails for this person, continue
                continue
    
    print(f"\n{'='*50}")
    print(f"📊 SEARCH COMPLETE!")
    print(f"✅ Total matches found: {matched_count}")
    if matched_count == 0:
        print(f"❌ No matches found!")
        print(f"💡 Tips:")
        print(f"   - Make sure face is clearly visible in reference image")
        print(f"   - Try lowering threshold (0.50) if face is similar but not exact")
        print(f"   - Check if face in video is visible and not blurry")
    print(f"{'='*50}")
    
    print(f"\n📊 ReID Complete | Total Matches: {matched_count}")

# ---- 2. PERSON WITHOUT ATTRIBUTE ----
def run_person_without_attribute_mode(cap, attribute, results_list):
    tracker  = CooldownTracker()
    frame_id = 0

    ATTR_YOLO_MAP = {
        "helmet":     "helmet",
        "cap":        "hat",
        "hat":        "hat",
        "mask":       "mask",
        "glasses":    "glasses",
        "sunglasses": "sunglasses",
        "bag":        "bag",
        "backpack":   "backpack",
        "vest":       "vest",
        "jacket":     "jacket",
        "uniform":    "uniform",
        "gloves":     "gloves",
        "shoe":       "shoes",
        "shoes":      "shoes",
        "gun":        "gun",
        "knife":      "knife",
    }
    yolo_attr = ATTR_YOLO_MAP.get(attribute, attribute)
    print(f"  Searching: person WITHOUT {attribute}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.35, imgsz=640)
        if res_p[0].boxes is None or len(res_p[0].boxes) == 0: continue

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()

        world_model.set_classes([yolo_attr])
        res_a = world_model(frame, conf=0.25, imgsz=640)
        a_boxes = []
        if res_a[0].boxes is not None and len(res_a[0].boxes) > 0:
            a_boxes = res_a[0].boxes.xyxy.cpu().numpy()

        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            if (px2-px1)*(py2-py1) < AREA_THRESHOLD: continue
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0: continue

            has_attr = False
            for a_box in a_boxes:
                ax1, ay1, ax2, ay2 = map(int, a_box)
                if attr_belongs_to_person([px1,py1,px2,py2],
                                          [ax1,ay1,ax2,ay2], attribute):
                    has_attr = True; break

            if not has_attr and attribute in HEAD_ATTRS:
                h_crop, w_crop = crop.shape[:2]
                head = crop[:int(h_crop*0.40), :]
                if head.size > 0:
                    world_model.set_classes([yolo_attr])
                    hr = world_model(head, conf=0.20, imgsz=320)
                    if hr[0].boxes is not None and len(hr[0].boxes) > 0:
                        has_attr = True

            if has_attr: continue

            label = f"person without {attribute}"
            key   = f"no_{attribute}_{px1//50}_{py1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, label, (0, 0, 255))
            img_path = os.path.join(SAVE_DIR, f"no_{attribute}_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"  🚨 {label} | frame={frame_id}")

            results_list.append({
                "object":     label,
                "attribute":  attribute,
                "violation":  True,
                "confidence": 0.85,
                "image_path": img_path,
                "bbox":       [px1, py1, px2, py2],
                "timestamp":  frame_id
            })

# ---- 3. COLOR + VEHICLE ----
def run_color_object_mode(cap, color_name, vehicle_word,
                          vehicle_info, results_list):
    tracker  = CooldownTracker()
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        for box in detect_vehicles_in_frame(frame, vehicle_info):
            x1, y1, x2, y2 = box
            if (x2-x1) < 15 or (y2-y1) < 15: continue

            full_crop = frame[y1:y2, x1:x2]
            if full_crop.size == 0: continue

            h, w = full_crop.shape[:2]
            if color_name in ("white", "silver"):
                body = full_crop[int(h*.05):int(h*.85),
                                 int(w*.15):int(w*.85)]
            else:
                body = full_crop[int(h*.10):int(h*.90),
                                 int(w*.10):int(w*.90)]
            body = body if body.size > 0 else full_crop

            if color_name:
                threshold = COLOR_THRESHOLDS.get(color_name.lower(), 0.15)
                cr = color_ratio(body, color_name)
                if cr < threshold: continue
                label = f"{color_name} {vehicle_word}"
                conf  = round(float(cr), 3)
            else:
                dom   = get_dominant_color_name(body)
                label = f"{dom} {vehicle_word}"
                conf  = 0.85

            key = f"cv_{x1//50}_{y1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            img_path = save_detection(
                frame, [x1,y1,x2,y2], label, frame_id, "color_veh")
            print(f"  ✅ {label} | frame={frame_id} | ratio={conf}")

            results_list.append({
                "object":     label,
                "color":      color_name or "any",
                "vehicle":    vehicle_word,
                "confidence": conf,
                "image_path": img_path,
                "bbox":       [x1, y1, x2, y2],
                "timestamp":  frame_id
            })

# ---- 4. PERSON + ATTRIBUTE ----
def run_person_attribute_mode(cap, attributes, results_list):
    tracker  = CooldownTracker()
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.35, imgsz=640)

        world_model.set_classes(attributes)
        res_a = world_model(frame, conf=0.30, imgsz=640)

        if res_p[0].boxes is None or len(res_p[0].boxes) == 0: continue
        if res_a[0].boxes is None or len(res_a[0].boxes) == 0: continue

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
        p_confs = res_p[0].boxes.conf.cpu().numpy()
        a_boxes = res_a[0].boxes.xyxy.cpu().numpy()
        a_clss  = res_a[0].boxes.cls.cpu().numpy()

        attr_det = [
            (list(map(int, ab)), attributes[int(ac)])
            for ab, ac in zip(a_boxes, a_clss)
        ]

        for pbox, _ in zip(p_boxes, p_confs):
            px1,py1,px2,py2 = map(int, pbox)
            if (px2-px1)*(py2-py1) < AREA_THRESHOLD: continue
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0: continue

            matched = [
                attr for a_box, attr in attr_det
                if attr_belongs_to_person([px1,py1,px2,py2], a_box, attr)
            ]
            if not matched: continue

            label = "person with " + " & ".join(set(matched))
            key   = f"pattr_{px1//50}_{py1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            print(f"  ✅ {label} | frame={frame_id}")
            results_list.append({
                "object":     label,
                "attributes": list(set(matched)),
                "confidence": 0.85,
                "image_path": save_detection(
                    frame,[px1,py1,px2,py2],
                    label,frame_id,"person_attr",(0,200,255)),
                "bbox":      [px1,py1,px2,py2],
                "timestamp": frame_id
            })

# ---- 5. PERSON + ATTRIBUTE + COLOR ----
def run_person_attribute_color_mode(cap, attribute, color_name, results_list):
    tracker   = CooldownTracker()
    frame_id  = 0
    yolo_attr = "shoes" if attribute == "shoe" else attribute

    print(f"  Searching: person with {attribute}"
          + (f" wearing {color_name}" if color_name else " (any color)"))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.35, imgsz=640)
        if res_p[0].boxes is None or len(res_p[0].boxes) == 0: continue

        world_model.set_classes([yolo_attr])
        res_a = world_model(frame, conf=0.25, imgsz=640)

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
        a_boxes = []
        if res_a[0].boxes is not None and len(res_a[0].boxes) > 0:
            a_boxes = res_a[0].boxes.xyxy.cpu().numpy()

        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            if (px2-px1)*(py2-py1) < AREA_THRESHOLD: continue
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0: continue

            attr_found = False
            matched_attr_boxes = []
            for a_box in a_boxes:
                ax1, ay1, ax2, ay2 = map(int, a_box)
                if attr_belongs_to_person([px1,py1,px2,py2],
                                          [ax1,ay1,ax2,ay2], attribute):
                    attr_found = True
                    matched_attr_boxes.append([ax1, ay1, ax2, ay2])

            if not attr_found and attribute in HEAD_ATTRS:
                h_crop, w_crop = crop.shape[:2]
                head = crop[:int(h_crop*0.40), :]
                if head.size > 0:
                    world_model.set_classes([yolo_attr])
                    hr = world_model(head, conf=0.20, imgsz=320)
                    if hr[0].boxes is not None and len(hr[0].boxes) > 0:
                        attr_found = True

            if not attr_found: continue

            detected_color = color_name
            if color_name:
                h, w = crop.shape[:2]
                if yolo_attr == "shoes":
                    if not matched_attr_boxes:
                        # Strict mode: if shoe bbox is not found, do not save.
                        continue
                    threshold = get_shoe_color_threshold(color_name)
                    best_shoe_ratio = 0.0
                    best_shoe_other = 0.0
                    for sb in matched_attr_boxes:
                        ax1, ay1, ax2, ay2 = sb
                        local_x1 = max(0, ax1 - px1)
                        local_y1 = max(0, ay1 - py1)
                        local_x2 = min(w, ax2 - px1)
                        local_y2 = min(h, ay2 - py1)
                        shoe_region = crop[local_y1:local_y2, local_x1:local_x2]
                        if shoe_region.size == 0:
                            continue
                        # Small padding captures full shoe when bbox is tight.
                        ph, pw = shoe_region.shape[:2]
                        pad_x = max(1, int(pw * 0.10))
                        pad_y = max(1, int(ph * 0.12))
                        ex1 = max(0, local_x1 - pad_x)
                        ey1 = max(0, local_y1 - pad_y)
                        ex2 = min(w, local_x2 + pad_x)
                        ey2 = min(h, local_y2 + pad_y)
                        expanded = crop[ey1:ey2, ex1:ex2]
                        if expanded.size > 0:
                            shoe_region = expanded
                        target_ratio = color_ratio(shoe_region, color_name)
                        other_best = 0.0
                        for cname in COLOR_RANGES:
                            if cname == color_name:
                                continue
                            other_best = max(other_best, color_ratio(shoe_region, cname))
                        if target_ratio > best_shoe_ratio:
                            best_shoe_ratio = target_ratio
                            best_shoe_other = other_best

                    # Requested color should be present and not clearly weaker than another color.
                    if best_shoe_ratio < threshold:
                        continue
                    if (best_shoe_ratio + 0.02) < best_shoe_other:
                        continue
                    conf = round(float(best_shoe_ratio), 3)
                else:
                    torso = crop[int(h*0.20):int(h*0.80),
                                 int(w*0.10):int(w*0.90)]
                    color_region = torso if torso.size > 0 else crop
                    threshold = COLOR_THRESHOLDS.get(color_name.lower(), 0.12)
                    cr = color_ratio(color_region, color_name)
                    if cr < threshold: continue
                    conf = round(float(cr), 3)
            else:
                h, w = crop.shape[:2]
                if yolo_attr == "shoes":
                    if not matched_attr_boxes:
                        continue
                    dom_counts = {}
                    for sb in matched_attr_boxes:
                        ax1, ay1, ax2, ay2 = sb
                        local_x1 = max(0, ax1 - px1)
                        local_y1 = max(0, ay1 - py1)
                        local_x2 = min(w, ax2 - px1)
                        local_y2 = min(h, ay2 - py1)
                        shoe_region = crop[local_y1:local_y2, local_x1:local_x2]
                        if shoe_region.size == 0:
                            continue
                        dom = get_dominant_color_name(shoe_region)
                        dom_counts[dom] = dom_counts.get(dom, 0) + 1
                    if not dom_counts:
                        continue
                    detected_color = max(dom_counts.items(), key=lambda kv: kv[1])[0]
                else:
                    torso = crop[int(h*0.20):int(h*0.80),
                                 int(w*0.10):int(w*0.90)]
                    color_region = torso if torso.size > 0 else crop
                    detected_color = get_dominant_color_name(color_region)
                conf = 0.85

            label = f"person with {attribute}"
            if detected_color and detected_color != "unknown":
                label += f" wearing {detected_color}"

            key = f"pac_{px1//50}_{py1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, label, (0, 200, 255))
            img_path = os.path.join(SAVE_DIR, f"attr_color_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"  ✅ {label} | frame={frame_id}")

            results_list.append({
                "object":     label,
                "attribute":  attribute,
                "color":      detected_color or "any",
                "confidence": conf,
                "image_path": img_path,
                "bbox":       [px1, py1, px2, py2],
                "timestamp":  frame_id
            })

# ---- 6. PERSON + VEHICLE + COLOR ----
def run_person_vehicle_color_mode(cap, vehicle_word, vehicle_info,
                                   color_name, results_list):
    tracker  = CooldownTracker()
    frame_id = 0
    print(f"  Searching: person on {vehicle_word} wearing {color_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.35, imgsz=640)
        if res_p[0].boxes is None or len(res_p[0].boxes) == 0: continue

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
        v_boxes = detect_vehicles_in_frame(frame, vehicle_info)
        if not v_boxes: continue

        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            if (px2-px1)*(py2-py1) < AREA_THRESHOLD: continue
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0: continue

            person_close = False
            prox = max(px2-px1, py2-py1) * 2.0
            for vbox in v_boxes:
                vx1,vy1,vx2,vy2 = vbox
                dist = np.sqrt(
                    ((px1+px2)/2-(vx1+vx2)/2)**2 +
                    ((py1+py2)/2-(vy1+vy2)/2)**2)
                if dist <= prox:
                    person_close = True; vehicle_bbox = vbox; break
            if not person_close: continue

            if color_name:
                h, w = crop.shape[:2]
                torso = crop[int(h*0.20):int(h*0.80),
                             int(w*0.10):int(w*0.90)]
                color_region = torso if torso.size > 0 else crop
                threshold = COLOR_THRESHOLDS.get(color_name.lower(), 0.12)
                if color_ratio(color_region, color_name) < threshold:
                    continue

            label = f"person on {vehicle_word}"
            if color_name: label += f" wearing {color_name}"

            key = f"pvc_{px1//50}_{py1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            ann = frame.copy()
            draw_box(ann, px1,py1,px2,py2, "person", (0,255,0))
            draw_box(ann, vehicle_bbox[0], vehicle_bbox[1],
                     vehicle_bbox[2], vehicle_bbox[3], vehicle_word, (255,100,0))
            cv2.putText(ann, label, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
            img_path = os.path.join(SAVE_DIR, f"veh_color_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"  ✅ {label} | frame={frame_id}")

            results_list.append({
                "object":       label,
                "vehicle":      vehicle_word,
                "color":        color_name or "any",
                "confidence":   0.85,
                "person_bbox":  [px1,py1,px2,py2],
                "vehicle_bbox": vehicle_bbox,
                "image_path":   img_path,
                "timestamp":    frame_id
            })

# ---- 7. ✅ PERSON WEARING COLOR (any clothes) ----
def run_color_attribute_mode(cap, object_class, color_name, results_list):
    """
    Detect: person wearing COLOR
    Works for: "person wearing red", "person wearing green shirt",
               "person in blue jacket", "person in yellow" etc.
    """
    tracker  = CooldownTracker()
    frame_id = 0
    world_model.set_classes([object_class])

    print(f"  Searching: {object_class} wearing {color_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        res = world_model(frame, conf=0.25, imgsz=640)
        if not res[0].boxes: continue

        for box in res[0].boxes.xyxy.cpu().numpy():
            x1,y1,x2,y2 = map(int, box)
            if (x2-x1)*(y2-y1) < AREA_THRESHOLD: continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            h,w = crop.shape[:2]

            if color_name and color_name.lower() != "any":
                threshold = COLOR_THRESHOLDS.get(color_name.lower(), 0.12)
                if color_name.lower() in {"white", "silver", "cream", "beige"}:
                    threshold = min(threshold, 0.16)

                # Region 1: Full body
                full_cr = color_ratio(crop, color_name)

                # Region 2: Torso (main clothes area)
                torso = crop[int(h*.15):int(h*.85), int(w*.10):int(w*.90)]
                torso_cr = color_ratio(torso, color_name) if torso.size > 0 else 0

                # Region 3: Upper body
                upper = crop[int(h*.10):int(h*.55), :]
                upper_cr = color_ratio(upper, color_name) if upper.size > 0 else 0

                # ✅ ANY region match kare toh detect karo
                if color_name.lower() in {"white", "silver", "cream", "beige"}:
                    best_cr = max(torso_cr, upper_cr, full_cr * 0.90)
                else:
                    best_cr = max(full_cr, torso_cr, upper_cr)
                if best_cr < threshold: continue
                conf  = round(float(best_cr), 3)
                label = f"person wearing {color_name}"
            else:
                # Any color — dominant color find karo
                torso = crop[int(h*.15):int(h*.85), int(w*.10):int(w*.90)]
                reg   = torso if torso.size > 0 else crop
                dom   = get_dominant_color_name(reg)
                label = f"person wearing {dom}"
                conf  = 0.85

            key = f"clothes_{x1//50}_{y1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            print(f"  ✅ {label} | frame={frame_id} | conf={conf}")
            results_list.append({
                "object":     label,
                "color":      color_name or "any",
                "confidence": conf,
                "image_path": save_detection(
                    frame,[x1,y1,x2,y2],label,
                    frame_id,"clothes",(255,165,0)),
                "bbox":      [x1,y1,x2,y2],
                "timestamp": frame_id
            })

# ---- 8. PERSON + VEHICLE PROXIMITY ----
def run_person_vehicle_mode(cap, vehicle_word, vehicle_info, results_list):
    tracker  = CooldownTracker()
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        # ✅ FIX 1: set_classes BEFORE calling world_model
        world_model.set_classes(["person"])
        pr = world_model(frame, conf=0.20, imgsz=640)
        if not pr[0].boxes: continue
        p_boxes = pr[0].boxes.xyxy.cpu().numpy()

        # ✅ FIX 2: Try YOLOv8n first, fallback to YOLOWorld for bike
        v_boxes = detect_vehicles_in_frame(
            frame, vehicle_info,
            conf_yolo=0.15,
            conf_world=0.15
        )

        # ✅ FIX 3: If still no boxes, try YOLOWorld with "motorcycle" / "bike"
        if not v_boxes:
            for world_label in ["motorcycle", "motorbike", "bike", "bicycle"]:
                world_model.set_classes([world_label])
                res_v = world_model(frame, conf=0.15, imgsz=640)
                if res_v[0].boxes is not None:
                    for b in res_v[0].boxes.xyxy.cpu().numpy():
                        v_boxes.append(list(map(int, b)))
                if v_boxes:
                    break

        if not v_boxes: continue

        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            if (px2-px1)*(py2-py1) < 150: continue  # ✅ lower threshold
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0: continue

            person_w = px2 - px1
            person_h = py2 - py1
            # ✅ FIX 4: Large proximity for overhead angle
            prox = max(person_w, person_h) * 5.0

            for vbox in v_boxes:
                vx1, vy1, vx2, vy2 = vbox
                overlap = iou([px1,py1,px2,py2], [vx1,vy1,vx2,vy2])
                dist = np.sqrt(
                    ((px1+px2)/2 - (vx1+vx2)/2)**2 +
                    ((py1+py2)/2 - (vy1+vy2)/2)**2
                )

                if dist > prox and overlap < 0.03:
                    continue

                key = f"pv_{px1//50}_{py1//50}"
                if tracker.should_skip(key, frame_id): continue
                tracker.update(key, frame_id)

                ann = frame.copy()
                draw_box(ann, px1, py1, px2, py2, "person", (0, 255, 0))
                draw_box(ann, vx1, vy1, vx2, vy2, vehicle_word, (255, 100, 0))
                path = os.path.join(SAVE_DIR, f"pv_{frame_id}.jpg")
                cv2.imwrite(path, ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
                print(f"  ✅ person+{vehicle_word} | frame={frame_id} | dist={dist:.1f} | iou={overlap:.3f}")

                results_list.append({
                    "object":       f"person with {vehicle_word}",
                    "vehicle":      vehicle_word,
                    "confidence":   0.85,
                    "person_bbox":  [px1, py1, px2, py2],
                    "vehicle_bbox": [vx1, vy1, vx2, vy2],
                    "image_path":   path,
                    "timestamp":    frame_id
                })
                break

# ---- 9. PERSON + COLOR HELMET ----
def run_person_helmet_color_mode(cap, color_name, results_list):
    tracker  = CooldownTracker()
    frame_id = 0
    print(f"  Searching: person with {color_name} helmet")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.35, imgsz=640)
        if res_p[0].boxes is None: continue

        world_model.set_classes(["helmet"])
        res_h = world_model(frame, conf=0.25, imgsz=640)
        if res_h[0].boxes is None: continue

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
        h_boxes = res_h[0].boxes.xyxy.cpu().numpy()

        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            if (px2-px1)*(py2-py1) < AREA_THRESHOLD: continue

            for hbox in h_boxes:
                hx1, hy1, hx2, hy2 = map(int, hbox)
                if not helmet_belongs_to_person(
                        [px1,py1,px2,py2], [hx1,hy1,hx2,hy2]):
                    continue

                helmet_crop = frame[hy1:hy2, hx1:hx2]
                if helmet_crop.size == 0: continue

                threshold = COLOR_THRESHOLDS.get(color_name.lower(), 0.12)
                if not detect_color_in_crop(helmet_crop, color_name, threshold):
                    continue

                label = f"person with {color_name} helmet"
                key   = f"helmet_color_{px1//50}_{py1//50}"
                if tracker.should_skip(key, frame_id): continue
                tracker.update(key, frame_id)

                ann = frame.copy()
                draw_box(ann, px1,py1,px2,py2, "person", (0,255,0))
                draw_box(ann, hx1,hy1,hx2,hy2,
                         f"{color_name} helmet", (255,0,0))
                img_path = os.path.join(SAVE_DIR, f"helmet_color_{frame_id}.jpg")
                cv2.imwrite(img_path, ann)
                print(f"  ✅ {label} | frame={frame_id}")

                results_list.append({
                    "object":       label,
                    "color":        color_name,
                    "confidence":   0.85,
                    "person_bbox":  [px1,py1,px2,py2],
                    "helmet_bbox":  [hx1,hy1,hx2,hy2],
                    "image_path":   img_path,
                    "timestamp":    frame_id
                })

# ---- 10. PERSON + COLOR BAG ----
def run_person_bag_color_mode(cap, color_name, bag_type, results_list):
    tracker  = CooldownTracker()
    frame_id = 0
    print(f"  Searching: person with {color_name} {bag_type}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.35, imgsz=640)

        world_model.set_classes([bag_type])
        res_b = world_model(frame, conf=0.30, imgsz=640)

        if res_p[0].boxes is None or res_b[0].boxes is None: continue

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
        b_boxes = res_b[0].boxes.xyxy.cpu().numpy()

        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            if (px2-px1)*(py2-py1) < AREA_THRESHOLD: continue

            for bbox in b_boxes:
                bx1, by1, bx2, by2 = map(int, bbox)
                if not bag_belongs_to_person(
                        [px1,py1,px2,py2], [bx1,by1,bx2,by2]):
                    continue

                bag_crop = frame[by1:by2, bx1:bx2]
                if bag_crop.size == 0: continue

                threshold = COLOR_THRESHOLDS.get(color_name.lower(), 0.12)
                if not detect_color_in_crop(bag_crop, color_name, threshold):
                    continue

                label = f"person with {color_name} {bag_type}"
                key   = f"bag_{px1//50}_{py1//50}"
                if tracker.should_skip(key, frame_id): continue
                tracker.update(key, frame_id)

                ann = frame.copy()
                draw_box(ann, px1,py1,px2,py2, "person", (0,255,0))
                draw_box(ann, bx1,by1,bx2,by2,
                         f"{color_name} {bag_type}", (255,0,0))
                img_path = os.path.join(SAVE_DIR, f"bag_color_{frame_id}.jpg")
                cv2.imwrite(img_path, ann)
                print(f"  ✅ {label} | frame={frame_id}")

                results_list.append({
                    "object":      label,
                    "color":       color_name,
                    "bag_type":    bag_type,
                    "confidence":  0.85,
                    "person_bbox": [px1,py1,px2,py2],
                    "bag_bbox":    [bx1,by1,bx2,by2],
                    "image_path":  img_path,
                    "timestamp":   frame_id
                })

# ---- 11. GENDER + COLOR ----
def run_gender_color_mode(cap, gender_target, color_name, results_list):
    tracker  = CooldownTracker()
    frame_id = 0
    print(f"  Searching: {gender_target} wearing {color_name}")
    world_model.set_classes(["person"])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        res = world_model(frame, conf=0.35, imgsz=640)
        if res[0].boxes is None: continue

        for box in res[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            if (x2-x1)*(y2-y1) < AREA_THRESHOLD: continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            face_found = False; face_crop = None
            try:
                faces = DeepFace.extract_faces(
                    crop, enforce_detection=False)
                if faces and len(faces) > 0:
                    face_img   = faces[0]["face"]
                    face_crop  = (face_img * 255).astype("uint8")
                    face_found = True
            except:
                face_found = False

            if not face_found: continue

            gender = predict_gender_deepface(face_crop)
            if gender != gender_target: continue

            h, w = crop.shape[:2]
            torso = crop[int(h*0.35):int(h*0.65),
                         int(w*0.25):int(w*0.75)]
            region = torso if torso.size > 0 else crop

            threshold = COLOR_THRESHOLDS.get(color_name.lower(), 0.12)
            if not detect_color_in_crop(region, color_name, threshold):
                continue

            label = f"{gender_target} wearing {color_name}"
            key   = f"gender_{x1//50}_{y1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            img_path = save_detection(
                frame, [x1,y1,x2,y2], label,
                frame_id, "gender", (255,0,255))
            print(f"  ✅ {label} | frame={frame_id}")

            results_list.append({
                "object":     label,
                "gender":     gender,
                "color":      color_name,
                "confidence": 0.85,
                "bbox":       [x1,y1,x2,y2],
                "image_path": img_path,
                "timestamp":  frame_id
            })

# ---- 12. TEXT SEARCH ----
def run_text_search_mode(cap, target_text, results_list):
    tracker  = CooldownTracker()
    frame_id = 0
    print(f"  Searching text: {target_text}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        results = reader.readtext(frame, detail=1)
        for (bbox, text, conf) in results:
            clean = re.sub(r'[^A-Z0-9]', '', text.upper())
            if len(clean) < 3: continue
            if target_text not in clean: continue

            pts = np.array(bbox).astype(int)
            x1, y1 = pts[0]; x2, y2 = pts[2]

            key = f"text_{x1//50}_{y1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            label = f"text: {clean}"
            ann   = frame.copy()
            draw_box(ann, x1, y1, x2, y2, label, (0,255,255))
            img_path = os.path.join(SAVE_DIR, f"text_{frame_id}.jpg")
            cv2.imwrite(img_path, ann)
            print(f"  ✅ {label} | frame={frame_id}")

            results_list.append({
                "object":        "text",
                "detected_text": clean,
                "target":        target_text,
                "confidence":    float(conf),
                "bbox":          [x1, y1, x2, y2],
                "image_path":    img_path,
                "timestamp":     frame_id
            })

# ---- 13. GENERIC YOLOWorld FALLBACK ----
def run_yoloworld_mode(cap, prompt, ref_feat, results_list):
    tracker  = CooldownTracker()
    frame_id = 0
    prompts  = [p.strip().lower() for p in prompt.split(",")]
    world_model.set_classes(prompts)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        res = world_model(frame, conf=0.30, imgsz=640)
        if not res[0].boxes: continue

        boxes = res[0].boxes.xyxy.cpu().numpy()
        clss  = res[0].boxes.cls.cpu().numpy()
        confs = res[0].boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, clss, confs):
            x1,y1,x2,y2 = map(int, box)
            label = prompts[int(cls_id)]
            if (x2-x1)*(y2-y1) < AREA_THRESHOLD: continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            if ref_feat is not None:
                feat = extract_features(crop)
                sim  = np.dot(ref_feat, feat) / (
                    np.linalg.norm(ref_feat)*np.linalg.norm(feat)+1e-8)
                if sim < 0.70: continue
                conf = float(sim)

            key = f"{label}_{x1//50}_{y1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            print(f"  ✅ {label} conf={float(conf):.2f} | frame={frame_id}")
            results_list.append({
                "object":     label,
                "confidence": float(conf),
                "image_path": save_detection(
                    frame,[x1,y1,x2,y2],
                    f"{label} {float(conf):.2f}",
                    frame_id, label),
                "bbox":      [x1,y1,x2,y2],
                "timestamp": frame_id
            })

# ================================
# MAIN API
# ================================
@app.post("/process")
async def process_video(req: Request):
    data = await req.json()

    video_url = data.get("fileUrl")
    prompt = data.get("prompt", "person")
    image_url = data.get("imageUrl")

    print(f"\n{'='*60}")
    print(f"🔍 Searching: {prompt}")
    print(f"{'='*60}")

    # ✅ FIXED: INITIALIZE VARIABLES
    ref_feat = None
    ref_img_bgr = None
    is_reid_mode = False

    # ================================
    # DOWNLOAD VIDEO
    # ================================
    try:
        r = requests.get(video_url, stream=True, timeout=10)
        with open("temp_video.mp4", "wb") as f:
            for chunk in r.iter_content(1024):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        return {"error": f"Video download failed: {e}"}

    cap = cv2.VideoCapture("temp_video.mp4")
    if not cap.isOpened():
        return {"error": "video open failed"}

    results_list = []

    # ================================
    # ✅ FIXED: HANDLE IMAGE (ReID)
    # ================================
    if image_url:
        try:
            print("📥 Downloading reference image...")
            ri = requests.get(image_url, stream=True, timeout=10)
            ri.raise_for_status()
            
            with open("temp_image.jpg", "wb") as f:
                for chunk in ri.iter_content(1024):
                    if chunk:
                        f.write(chunk)
            
            img = cv2.imread("temp_image.jpg")
            if img is None:
                print("⚠️ Failed to read image")
            else:
                print(f"✅ Image loaded: {img.shape}")
                ref_img_bgr = img.copy()
                
                # ✅ PERSON DETECTION FROM IMAGE
                print("🔍 Detecting person in reference image...")
                world_model.set_classes(["person"])
                res_ref = world_model(img, conf=0.20, imgsz=640)
                
                # ✅ FIXED: Initialize before if-else
                bx1, by1, bx2, by2 = None, None, None, None
                
                if (res_ref[0].boxes is not None and 
                    len(res_ref[0].boxes) > 0):
                    
                    boxes = res_ref[0].boxes.xyxy.cpu().numpy()
                    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                    best_idx = int(np.argmax(areas))
                    bx1, by1, bx2, by2 = map(int, boxes[best_idx])
                    
                    print(f"✅ Found person at: ({bx1},{by1}) - ({bx2},{by2})")
                    
                    h_img, w_img = img.shape[:2]
                    pad = 20
                    
                    bx1 = max(0, bx1 - pad)
                    by1 = max(0, by1 - pad)
                    bx2 = min(w_img, bx2 + pad)
                    by2 = min(h_img, by2 + pad)
                    
                    person_crop = img[by1:by2, bx1:bx2]
                    
                    if person_crop.size > 0:
                        # ✅ CONSISTENT FEATURE EXTRACTION
                        ref_feat = extract_person_features(
                            img, [bx1, by1, bx2, by2]
                        )
                        
                        if ref_feat is not None:
                            print("✅ Features extracted from person crop")
                            is_reid_mode = True
                        else:
                            # Fallback
                            ref_feat = extract_features(person_crop)
                            print("⚠️ Using simple features")
                            is_reid_mode = True
                else:
                    print("⚠️ No person detected, using full image")
                    ref_feat = extract_features(img)
                    is_reid_mode = True
                    
        except Exception as e:
            print(f"❌ Image processing failed: {e}")
            ref_feat = None
            is_reid_mode = False

    # ================================
    # ✅ FIXED: EARLY MODE DETECTION
    # ================================
    parsed = parse_prompt(prompt)
    mode = parsed.get("mode", "yoloworld")

    # ✅ EARLY ReID DETECTION
    reid_triggers = [
        "find", "match", "same", "identify", "track", 
        "person in image", "match person"
    ]
    is_user_asking_for_reid = any(
        trigger in prompt.lower() for trigger in reid_triggers
    )

    # ✅ AUTO-SWITCH TO ReID IF CONDITIONS MET
    if is_reid_mode and is_user_asking_for_reid:
        mode = "person_reid"
        parsed["mode"] = "person_reid"
        print("🔄 AUTO-SWITCHED TO REID MODE (Image + prompt matched)")
    elif is_reid_mode and "person" in prompt.lower():
        mode = "person_reid"
        parsed["mode"] = "person_reid"
        print("🔄 REID MODE ENABLED (Person + image provided)")

    print(f"\n📋 Final Mode: {mode}")
    print(f"📊 Has ReID Features: {ref_feat is not None}")
    print(f"📸 Has Reference Image: {ref_img_bgr is not None}")

    # ================================
    # MODE ROUTING
    # ================================
    if mode == "plate":
        run_plate_mode(cap, parsed["plate_info"], results_list)

    elif mode == "person_without_attribute":
        run_person_without_attribute_mode(
            cap, parsed["attribute"], results_list
        )

    elif mode == "person_helmet_color":
        run_person_helmet_color_mode(
            cap, parsed["color"], results_list
        )

    elif mode == "person_bag_color":
        run_person_bag_color_mode(
            cap, parsed["color"], parsed["bag_type"], results_list
        )

    elif mode == "text_search":
        run_text_search_mode(
            cap, parsed["target_text"], results_list
        )

    elif mode == "gender_color":
        run_gender_color_mode(
            cap, parsed["gender"], parsed["color"], results_list
        )

    elif mode == "person_attribute_color":
        run_person_attribute_color_mode(
            cap, parsed["attribute"], parsed.get("color"), results_list
        )

    elif mode == "person_vehicle_color":
        run_person_vehicle_color_mode(
            cap,
            parsed["vehicle_word"],
            parsed["vehicle_info"],
            parsed["color"],
            results_list,
        )

    elif mode == "color_attribute":
        run_color_attribute_mode(
            cap, parsed["object"], parsed["color"], results_list
        )

    elif mode == "color_object":
        run_color_object_mode(
            cap,
            parsed["color"],
            parsed["vehicle_word"],
            parsed["vehicle_info"],
            results_list,
        )

    elif mode == "person_attribute":
        run_person_attribute_mode(
            cap, parsed["attributes"], results_list
        )

    elif mode == "person_vehicle":
        run_person_vehicle_mode(
            cap,
            parsed["vehicle_word"],
            parsed["vehicle_info"],
            results_list,
        )

    elif mode == "person_reid":

        FACE_MATCH_THRESHOLD = 0.55
        # ✅ FIXED: Better ReID with stricter threshold
        run_person_reid_mode(
            cap, 
            ref_feat, 
            results_list, 
            ref_img_bgr=ref_img_bgr,
            threshold=FACE_MATCH_THRESHOLD  # ✅ STRICTER THRESHOLD
        )

    else:
        run_yoloworld_mode(
            cap, prompt, ref_feat, results_list
        )

    # ================================
    # FINAL PROCESSING
    # ================================
    cap.release()

    filtered_results, removed_results = filter_results_by_prompt(
        results_list, parsed, prompt
    )

    print(f"\n📊 Results found (raw): {len(results_list)}")
    print(f"🧹 Removed by strict prompt filter: {len(removed_results)}")
    print(f"✅ Results kept: {len(filtered_results)}")
    print(f"{'='*60}\n")

    return {
        "results": filtered_results,
        "mode": mode,
        "parsed_prompt": parsed,
        "total_found": len(filtered_results),
    }