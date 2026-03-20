
from fastapi import FastAPI, Request
from ultralytics import YOLO, YOLOWorld
import cv2
import os
import torch
import numpy as np
import requests
import easyocr
import re
from difflib import SequenceMatcher
from torchvision import models, transforms
from PIL import Image


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

FRAME_SKIP     = 3          # check more frames for plates
AREA_THRESHOLD = 1500
BLUR_THRESHOLD = 30
COOLDOWN       = 25

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])




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

# OCR common substitution pairs
OCR_SUBS = {
    '0':'O','O':'0','1':'I','I':'1','8':'B','B':'8',
    '5':'S','S':'5','2':'Z','Z':'2','6':'G','G':'6',
    '4':'A','D':'0',
}

PLATE_FULL_RE = re.compile(
    r'^([A-Z]{2})\s?(\d{1,2})\s?([A-Z]{1,3})\s?(\d{1,4})$'
)

# ================================
# PLATE UTILITIES
# ================================

def clean_plate(text):
    if not text:
        return ""
    return re.sub(r'[^A-Z0-9]', '', text.upper())





def ocr_variants(text):
    """Generate all single-substitution OCR variants."""
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
    """
    Auto-detect what kind of plate search to do.
    Works for ANY state — not just MP.
    Returns type + all needed fields for matching.
    """
    raw   = prompt.strip().upper()
    clean = re.sub(r'[^A-Z0-9]', '', raw)
    parts = raw.split()

    # ---- 1. Only state code  e.g. "MP", "UP", "DL"
    if clean in INDIA_STATE_CODES and len(clean) == 2:
        return {
            "type":  "state_only",
            "state": clean,
            "desc":  f"All vehicles from {INDIA_STATE_CODES[clean]}"
        }

    # ---- 2. State + RTO  e.g. "MP09", "UP32", "MH04"
    m = re.match(r'^([A-Z]{2})\s?(\d{1,2})$', raw)
    if m and m.group(1) in INDIA_STATE_CODES:
        st = m.group(1)
        rt = m.group(2).zfill(2)
        return {
            "type":   "state_rto",
            "state":  st,
            "rto":    rt,
            "prefix": st + rt,
            "desc":   f"RTO {st}-{rt} ({INDIA_STATE_CODES[st]})"
        }

    # ---- 3. Full plate
    parsed = parse_plate_string(raw)
    if parsed and parsed["state"] in INDIA_STATE_CODES:
        return {
            "type":   "full_plate",
            "parsed": parsed,
            "desc":   f"Exact plate {parsed['full']}"
        }

    # ---- 4. State + last digits (half & half)  e.g. "MP 7785"
    if len(parts) == 2:
        st_part  = re.sub(r'[^A-Z]', '', parts[0])
        num_part = re.sub(r'[^0-9]', '', parts[1])
        if st_part in INDIA_STATE_CODES and num_part:
            return {
                "type":   "state_number",
                "state":  st_part,
                "number": num_part,
                "desc":   f"State={st_part} AND ends with {num_part}"
            }

    # ---- 5. Half plate: series+number only  e.g. "ZP7785", "7785"
    if any(c.isdigit() for c in clean) and clean not in INDIA_STATE_CODES:
        return {
            "type":    "half_plate",
            "partial": clean,
            "desc":    f"Plate contains '{clean}'"
        }

    return {"type": "not_plate"}


def is_plate_query(prompt: str) -> bool:
    return detect_plate_prompt_type(prompt)["type"] != "not_plate"


# ================================
# SMART PLATE MATCHING
# ================================

def smart_plate_match(plate_info: dict, ocr_plate: str) -> bool:
    """
    Match detected OCR text against any prompt type.
    Works for all states — MP, UP, DL, MH, RJ, GJ... all of them.
    """
    plate = clean_plate(ocr_plate)
    if not plate or len(plate) < 2:
        return False

    ptype = plate_info["type"]
    plate_vars = ocr_variants(plate)

    # --------------------------------------------------
    # STATE ONLY — plate must start with state code
    # Works for ANY state: MP, UP, DL, MH, etc.
    # --------------------------------------------------
    if ptype == "state_only":
        state = plate_info["state"]
        # Direct check
        for v in plate_vars:
            if v.startswith(state):
                return True
        # Fuzzy on first 2 chars (handles OCR mangled state code)
        if len(plate) >= 2:
            if SequenceMatcher(None, state, plate[:2]).ratio() >= 0.80:
                return True
        return False

    # --------------------------------------------------
    # STATE + RTO — plate must start with state+rto
    # e.g. "MP04" matches MP04AB1234, MP04ZP7785
    # --------------------------------------------------
    elif ptype == "state_rto":
        prefix = plate_info["prefix"]   # e.g. "MP04"
        for v in plate_vars:
            if v.startswith(prefix):
                return True
        if len(plate) >= 4:
            if SequenceMatcher(None, prefix, plate[:4]).ratio() >= 0.78:
                return True
        # Also try: state matches first 2 AND rto matches next 2
        if len(plate) >= 4:
            state_match = SequenceMatcher(
                None, plate_info["state"], plate[:2]).ratio() >= 0.80
            rto_match   = SequenceMatcher(
                None, plate_info["rto"], plate[2:4]).ratio() >= 0.80
            if state_match and rto_match:
                return True
        return False

    # --------------------------------------------------
    # FULL PLATE — exact / substring / fuzzy
    # --------------------------------------------------
    elif ptype == "full_plate":
        target = plate_info["parsed"]["full"]
        target_vars = ocr_variants(target)

        # Exact
        if plate == target:
            return True
        # Variant exact
        for tv in target_vars:
            if plate == tv:
                return True
        # Substring (partial visibility of plate)
        for tv in target_vars:
            if tv in plate or plate in tv:
                return True
        # Fuzzy ratio
        if SequenceMatcher(None, target, plate).ratio() >= 0.72:
            return True
        return False

    # --------------------------------------------------
    # STATE + NUMBER  e.g. "MP 7785"
    # Both must match independently
    # --------------------------------------------------
    elif ptype == "state_number":
        state  = plate_info["state"]
        number = plate_info["number"]

        state_ok = False
        num_ok   = False

        for v in plate_vars:
            if v.startswith(state):
                state_ok = True
            if number in v:
                num_ok = True

        if not state_ok and len(plate) >= 2:
            state_ok = SequenceMatcher(
                None, state, plate[:2]).ratio() >= 0.80

        if not num_ok:
            # Check last N chars
            tail = plate[-len(number):]
            num_ok = SequenceMatcher(None, number, tail).ratio() >= 0.80

        return state_ok and num_ok

    # --------------------------------------------------
    # HALF PLATE — partial substring in plate
    # e.g. "ZP7785" or just "7785"
    # --------------------------------------------------
    elif ptype == "half_plate":
        partial = plate_info["partial"]
        partial_vars = ocr_variants(partial)

        for pv in partial_vars:
            if pv in plate:
                return True
            for v in plate_vars:
                if pv in v:
                    return True

        # Fuzzy ratio
        if SequenceMatcher(None, partial, plate).ratio() >= 0.75:
            return True

        # Sliding window for short partials
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
    """
    Apply multiple enhancement passes to maximize OCR accuracy.
    Returns list of enhanced versions to try.
    """
    versions = []
    h, w = crop.shape[:2]

    # Upscale if small
    scale = max(1.0, 200 / max(w, 1))
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Version 1: Otsu threshold
    _, v1 = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(v1)

    # Version 2: Inverted Otsu (for dark plates)
    versions.append(cv2.bitwise_not(v1))

    # Version 3: CLAHE + Gaussian
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    v3    = clahe.apply(gray)
    v3    = cv2.GaussianBlur(v3, (3, 3), 0)
    versions.append(v3)

    # Version 4: Sharpened
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    v4     = cv2.filter2D(gray, -1, kernel)
    versions.append(v4)

    return versions





def perform_ocr_plate(frame, box):
    """
    Try multiple enhancement versions and return best OCR result.
    Joins multi-part OCR results (e.g. "MP 04" + "ZP" + "7785").
    """
    x1, y1, x2, y2 = map(int, box)
    # Add padding around plate
    pad = 8
    x1  = max(0, x1-pad); y1 = max(0, y1-pad)
    x2  = min(frame.shape[1], x2+pad)
    y2  = min(frame.shape[0], y2+pad)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return ""

    enhanced_versions = enhance_plate_crop(crop)
    best_result = ""
    best_score  = 0

    for enhanced in enhanced_versions:
        # Try full join (all OCR parts concatenated)
        results = reader.readtext(
            enhanced, detail=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        )
        if not results:
            continue

        # Join all detected text parts
        combined = clean_plate("".join(results))

        # Score: longer valid plate = better
        # A valid Indian plate has 6-10 alphanumeric chars
        if 5 <= len(combined) <= 12:
            score = len(combined)
            # Bonus if starts with known state code
            if len(combined) >= 2 and combined[:2] in INDIA_STATE_CODES:
                score += 5
            if score > best_score:
                best_score  = score
                best_result = combined

        # Also try individual parts (sometimes OCR splits weirdly)
        for part in results:
            p = clean_plate(part)
            if len(p) >= 4:
                score = len(p)
                if len(p) >= 2 and p[:2] in INDIA_STATE_CODES:
                    score += 5
                if score > best_score:
                    best_score  = score
                    best_result = p

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
}

# ================================
# VEHICLE ALIASES
# ================================

VEHICLE_ALIASES = {
    "car":           {"strategy": "yolo_class", "class_ids": [2]},
    "truck":         {"strategy": "yolo_class", "class_ids": [7]},
    "bus":           {"strategy": "yolo_class", "class_ids": [5]},
    "van":           {"strategy": "yolo_class", "class_ids": [2, 7]},
    "bike":          {"strategy": "yolo_class", "class_ids": [1, 3]},
    "bicycle":       {"strategy": "yolo_class", "class_ids": [1]},
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
    "glasses", "sunglasses", "gloves", "shoes",
]

HEAD_ATTRS   = {"helmet", "cap", "hat", "mask", "glasses", "sunglasses"}
BODY_ATTRS   = {"bag", "backpack", "jacket", "vest", "uniform", "gloves", "shoes"}
WEAPON_ATTRS = {"weapon", "gun", "knife"}


# ================================
# GENERAL UTILITIES
# ================================

def extract_features(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        return resnet(img).flatten().numpy()


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


def detect_color_in_crop(crop, color_name, threshold=0.30):
    return color_ratio(crop, color_name) > threshold


def get_dominant_color_name(crop):
    best, best_r = "unknown", 0.0
    for cname in COLOR_RANGES:
        r = color_ratio(crop, cname)
        if r > best_r:
            best, best_r = cname, r
    return best


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
# PROMPT ROUTER - UPDATED
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

    # Check plate FIRST
    plate_info = detect_plate_prompt_type(prompt.strip())
    if plate_info["type"] != "not_plate":
        return {"mode": "plate", "plate_info": plate_info, "prompt": prompt}

    detected_color   = next((c for c in COLOR_RANGES if c in p), None)
    any_color        = bool(re.search(r'\bany[\s_]?colou?r\b', p))
    veh_word, v_info = find_vehicle_in_prompt(p)

    # ---- NEW: Person + Attribute + Color ----
    # e.g., "person with helmet wearing red"
    if "person" in p and "with" in p and ("wearing" in p or " in " in p):
        m = re.search(
            r'person\s+with\s+(\w+)\s+(?:wear(?:ing)?|in)\s+(\w+)',
            p
        )
        if m:
            attr = m.group(1).strip()
            color = m.group(2).strip()
            if attr in PERSON_ATTRIBUTE_KEYWORDS and color in COLOR_RANGES:
                return {
                    "mode": "person_attribute_color",
                    "attribute": attr,
                    "color": color,
                    "prompt": prompt
                }

        
       





    # ---- NEW: Person + Vehicle + Color ----
    # e.g., "person on bike wearing red"
    if "person" in p and veh_word and ("wearing" in p or " in " in p):
        m = re.search(
            r'person\s+(?:on|with|riding)\s+' + re.escape(veh_word) + r'\s+(?:wear(?:ing)?|in)\s+(\w+)',
            p
        )
        if m:
            color = m.group(1).strip()
            if color in COLOR_RANGES:
                return {
                    "mode": "person_vehicle_color",
                    "vehicle_word": veh_word,
                    "vehicle_info": v_info,
                    "color": color,
                    "prompt": prompt
                }

    m = re.search(
        r'person\s+(?:wear(?:ing)?|in|with)\s+(\w+)\s+'
        r'(?:cloth(?:es|ing)|dress|shirt|jacket|top|outfit)', p
    )
    if m:
        return {"mode": "color_attribute", "object": "person",
                "color": m.group(1), "prompt": prompt}

    if (detected_color or any_color) and veh_word:
        return {"mode": "color_object", "color": detected_color,
                "vehicle_word": veh_word, "vehicle_info": v_info,
                "prompt": prompt}

    if "person" in p:
        found = [kw for kw in PERSON_ATTRIBUTE_KEYWORDS if kw in p]
        if found:
            return {"mode": "person_attribute",
                    "attributes": found, "prompt": prompt}
        if veh_word:
            return {"mode": "person_vehicle",
                    "vehicle_word": veh_word, "vehicle_info": v_info,
                    "prompt": prompt}

    return {"mode": "yoloworld", "prompt": prompt}


# ================================
# VEHICLE DETECTION HELPER
# ================================

def detect_vehicles_in_frame(frame, vehicle_info,
                              conf_yolo=0.55, conf_world=0.45):
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
    return (px1<=hcx<=px2 and hcy<=py1+ph*0.38 and
            hy1>=py1-30 and iou(p_box,h_box)>0.04 and
            (hx2-hx1)<pw*0.70)


def bag_belongs_to_person(p_box, b_box):
    px1,py1,px2,py2=p_box
    cx=(b_box[0]+b_box[2])/2; cy=(b_box[1]+b_box[3])/2
    return px1<=cx<=px2 and py1<=cy<=py2


def weapon_belongs_to_person(p_box, w_box):
    px1,py1,px2,py2=p_box
    ew=(px2-px1)*0.3; eh=(py2-py1)*0.3
    cx=(w_box[0]+w_box[2])/2; cy=(w_box[1]+w_box[3])/2
    return (px1-ew<=cx<=px2+ew) and (py1-eh<=cy<=py2+eh)


def attr_belongs_to_person(p_box, a_box, attr_name):
    if attr_name in HEAD_ATTRS:    return helmet_belongs_to_person(p_box, a_box)
    elif attr_name in WEAPON_ATTRS: return weapon_belongs_to_person(p_box, a_box)
    else:                           return bag_belongs_to_person(p_box, a_box)


# ================================
# DETECTION MODES
# ================================

# ---- 1. LICENSE PLATE — DIRECT FRAME SCAN (KEY FIX) ----

def run_plate_mode(cap, plate_info, results_list):
    """
    STRATEGY — 3 parallel pipelines, all run every frame:

    Pipeline A (direct scan):
      plate_model directly on full frame → no vehicle dependency
      Catches plates even when vehicle is missed by car_model

    Pipeline B (vehicle-guided):
      car_model → find vehicles → plate_model on each vehicle crop
      More precise, lower false positives

    Pipeline C (multi-scale):
      Resize frame to 2x → plate_model
      Catches small/far plates

    All 3 pipelines feed into same smart_plate_match() function.
    Any match from any pipeline saves the frame.
    """
    tracker  = CooldownTracker(cooldown=25)
    frame_id = 0
    ptype    = plate_info["type"]

    print(f"  Plate type: {ptype} | {plate_info.get('desc','')}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        found_plates = []  # list of (plate_box, ocr_text)

        # =====================================================
        # PIPELINE A: Direct full-frame plate scan
        # Lowest conf (0.15) to catch everything
        # =====================================================
        direct_results = plate_model(frame, conf=0.15, imgsz=1280)
        if direct_results[0].boxes is not None:
            for pbox in direct_results[0].boxes.xyxy.cpu().numpy():
                px1,py1,px2,py2 = map(int, pbox)
                ocr = perform_ocr_plate(frame, [px1,py1,px2,py2])
                if ocr:
                    found_plates.append(([px1,py1,px2,py2], ocr, None))

        # =====================================================
        # PIPELINE B: Vehicle-guided (car → plate)
        # =====================================================
        car_results = car_model(frame, classes=[2,3,5,7], conf=0.30)
        if car_results[0].boxes is not None:
            for cbox in car_results[0].boxes.xyxy.cpu().numpy():
                cx1,cy1,cx2,cy2 = map(int, cbox)
                car_crop = frame[cy1:cy2, cx1:cx2]
                if car_crop.size == 0:
                    continue

                plate_in_car = plate_model(car_crop, conf=0.15)
                if plate_in_car[0].boxes is None:
                    continue

                for pbox in plate_in_car[0].boxes.xyxy.cpu().numpy():
                    px1,py1,px2,py2 = map(int, pbox)
                    # Convert to full-frame coords
                    px1+=cx1; py1+=cy1; px2+=cx1; py2+=cy1
                    ocr = perform_ocr_plate(frame, [px1,py1,px2,py2])
                    if ocr:
                        found_plates.append((
                            [px1,py1,px2,py2], ocr,
                            [cx1,cy1,cx2,cy2]
                        ))

        # =====================================================
        # PIPELINE C: Upscaled frame scan (catches small plates)
        # =====================================================
        h, w = frame.shape[:2]
        if w <= 1280:  # only upscale if not already large
            upscaled = cv2.resize(frame, (w*2, h*2),
                                  interpolation=cv2.INTER_LINEAR)
            up_results = plate_model(upscaled, conf=0.20, imgsz=1280)
            if up_results[0].boxes is not None:
                for pbox in up_results[0].boxes.xyxy.cpu().numpy():
                    # Scale coords back to original
                    px1,py1,px2,py2 = map(int, pbox)
                    px1//=2; py1//=2; px2//=2; py2//=2
                    ocr = perform_ocr_plate(frame, [px1,py1,px2,py2])
                    if ocr:
                        found_plates.append(([px1,py1,px2,py2], ocr, None))

        # =====================================================
        # Match all found plates against prompt
        # =====================================================
        for plate_box, ocr_text, car_box in found_plates:
            if not smart_plate_match(plate_info, ocr_text):
                continue

            # Dedup by position
            key = f"plate_{plate_box[0]//60}_{plate_box[1]//60}"
            if tracker.should_skip(key, frame_id):
                continue
            tracker.update(key, frame_id)

            # State name for label
            state_name = ""
            if len(ocr_text) >= 2 and ocr_text[:2] in INDIA_STATE_CODES:
                state_name = INDIA_STATE_CODES[ocr_text[:2]]

            label = ocr_text
            if state_name:
                label += f" ({state_name})"

            # Draw annotation
            ann = frame.copy()
            if car_box:
                draw_box(ann, *car_box, "vehicle", (255,200,0))
            draw_box(ann, *plate_box, label, (0,255,0))

            img_path = os.path.join(SAVE_DIR, f"plate_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])

            results_list.append({
                "object":       "license_plate",
                "plate":        ocr_text,
                "bbox":         plate_box if plate_box else [],
                "state":        ocr_text[:2] if len(ocr_text) >= 2 else "",
                "state_name":   state_name,
                "match_type":   ptype,
                "image_path":   img_path,
                "plate_bbox":   plate_box,
                "vehicle_bbox": car_box,
                "timestamp":    frame_id
            })




# ---- 2. COLOR + VEHICLE ----

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

            # ✅ Sirf minimum size check — bahut chhoti cheez skip
            if (x2-x1) < 20 or (y2-y1) < 20: continue

            full_crop = frame[y1:y2, x1:x2]
            if full_crop.size == 0: continue

            h, w = full_crop.shape[:2]
            body = full_crop[int(h*.15):int(h*.80),
                             int(w*.10):int(w*.90)]
            body = body if body.size > 0 else full_crop

            # ✅ COLOR MATCH — sirf yahi condition matter karti hai
            if color_name:
                cr = color_ratio(body, color_name)
                if cr < 0.20: continue   # color nahi mila → skip
                label = f"{color_name} {vehicle_word}"
                conf  = round(float(cr), 3)
            else:
                dom   = get_dominant_color_name(body)
                label = f"{dom} {vehicle_word}"
                conf  = 0.85

            # ✅ Cooldown
            key = f"cv_{x1//50}_{y1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            # ✅ MATCH HUA — tabhi save karo
            img_path = save_detection(
                frame, [x1,y1,x2,y2], label, frame_id, "color_veh")

            print(f"  ✅ {label} saved | frame={frame_id} | color_ratio={conf}")

            results_list.append({
                "object":     label,
                "color":      color_name or "any",
                "vehicle":    vehicle_word,
                "confidence": conf,
                "image_path": img_path,
                "bbox":       [x1, y1, x2, y2],
                "timestamp":  frame_id
            })


# ---- 3. PERSON + ATTRIBUTE ----

def run_person_attribute_mode(cap, attributes, results_list):
    tracker  = CooldownTracker()
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.40, imgsz=640)

        world_model.set_classes(attributes)
        res_a = world_model(frame, conf=0.35, imgsz=640)

        if (res_p[0].boxes is None or len(res_p[0].boxes) == 0 or
                res_a[0].boxes is None or len(res_a[0].boxes) == 0):
            continue

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
            if crop.size == 0 or is_blurry(crop): continue

            matched = [
                attr for a_box, attr in attr_det
                if attr_belongs_to_person([px1,py1,px2,py2], a_box, attr)
            ]
            if not matched: continue

            label = "person with " + " & ".join(set(matched))
            key   = f"pattr_{px1//50}_{py1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            results_list.append({
                "object":     label,
                "attributes": list(set(matched)),
                "image_path": save_detection(
                    frame,[px1,py1,px2,py2],
                    label,frame_id,"person_attr",(0,200,255)),
                "bbox":      [px1,py1,px2,py2],
                "timestamp": frame_id
            })


# ---- 3.5 NEW: PERSON + ATTRIBUTE + COLOR ----

def run_person_attribute_color_mode(cap, attribute, color_name, results_list):
    """🎯 Detect: Person with ATTRIBUTE + wearing COLOR"""
    tracker  = CooldownTracker()
    frame_id = 0
    
    print(f"  Searching: person with {attribute} wearing {color_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        # Detect person
        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.40, imgsz=640)
        if res_p[0].boxes is None or len(res_p[0].boxes) == 0:
            continue

        # Detect attribute
        world_model.set_classes([attribute])
        res_a = world_model(frame, conf=0.35, imgsz=640)
        if res_a[0].boxes is None or len(res_a[0].boxes) == 0:
            continue

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
        a_boxes = res_a[0].boxes.xyxy.cpu().numpy()

        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            if (px2 - px1) * (py2 - py1) < AREA_THRESHOLD: continue
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0 or is_blurry(crop): continue

            # Check if attribute belongs to person
            attr_found = False
            for a_box in a_boxes:
                ax1, ay1, ax2, ay2 = map(int, a_box)
                if attr_belongs_to_person([px1, py1, px2, py2], [ax1, ay1, ax2, ay2], attribute):
                    attr_found = True
                    break

            if not attr_found: continue

            # Check color of person's clothes
            h, w = crop.shape[:2]
            torso = crop[int(h*0.25):int(h*0.75), int(w*0.15):int(w*0.85)]
            color_region = torso if torso.size > 0 else crop

            if not detect_color_in_crop(color_region, color_name, threshold=0.15):
                continue

            # ✅ ALL CONDITIONS MET
            label = f"person with {attribute} wearing {color_name}"
            key = f"attr_color_{px1//50}_{py1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, label, (0, 200, 255))
            img_path = os.path.join(SAVE_DIR, f"attr_color_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])

            results_list.append({
                "object": label,
                "attribute": attribute,
                "color": color_name,
                "image_path": img_path,
                "bbox": [px1, py1, px2, py2],
                "timestamp": frame_id
            })


# ---- 3.6 NEW: PERSON + VEHICLE + COLOR ----

def run_person_vehicle_color_mode(cap, vehicle_word, vehicle_info, color_name, results_list):
    """🎯 Detect: Person with VEHICLE + wearing COLOR"""
    tracker  = CooldownTracker()
    frame_id = 0
    
    print(f"  Searching: person on {vehicle_word} wearing {color_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        # Detect person
        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.35, imgsz=640)
        if res_p[0].boxes is None or len(res_p[0].boxes) == 0:
            continue

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()

        # Detect vehicles
        v_boxes = detect_vehicles_in_frame(frame, vehicle_info)
        if not v_boxes: continue

        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            if (px2 - px1) * (py2 - py1) < AREA_THRESHOLD: continue
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0 or is_blurry(crop): continue

            # Check proximity to vehicle
            person_close = False
            prox = max(px2 - px1, py2 - py1) * 2.0
            
            for vbox in v_boxes:
                vx1, vy1, vx2, vy2 = vbox
                dist = np.sqrt(
                    ((px1+px2)/2 - (vx1+vx2)/2)**2 +
                    ((py1+py2)/2 - (vy1+vy2)/2)**2
                )
                if dist <= prox:
                    person_close = True
                    vehicle_bbox = vbox
                    break

            if not person_close: continue

            # Check color
            h, w = crop.shape[:2]
            torso = crop[int(h*0.25):int(h*0.75), int(w*0.15):int(w*0.85)]
            color_region = torso if torso.size > 0 else crop

            if not detect_color_in_crop(color_region, color_name, threshold=0.15):
                continue

            # ✅ ALL CONDITIONS MET
            label = f"person on {vehicle_word} wearing {color_name}"
            key = f"veh_color_{px1//50}_{py1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, "person", (0, 255, 0))
            draw_box(ann, vehicle_bbox[0], vehicle_bbox[1], vehicle_bbox[2], vehicle_bbox[3], vehicle_word, (255, 100, 0))
            cv2.putText(ann, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            img_path = os.path.join(SAVE_DIR, f"veh_color_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])

            results_list.append({
                "object": label,
                "vehicle": vehicle_word,
                "color": color_name,
                "person_bbox": [px1, py1, px2, py2],
                "vehicle_bbox": vehicle_bbox,
                "image_path": img_path,
                "timestamp": frame_id
            })


# ---- 4. PERSON WEARING COLOR CLOTHES ----

def run_color_attribute_mode(cap, object_class, color_name, results_list):
    tracker  = CooldownTracker()
    frame_id = 0
    world_model.set_classes([object_class])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        res = world_model(frame, conf=0.35, imgsz=640)
        if not res[0].boxes: continue

        for box in res[0].boxes.xyxy.cpu().numpy():
            x1,y1,x2,y2 = map(int, box)
            if (x2-x1)*(y2-y1) < AREA_THRESHOLD: continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or is_blurry(crop): continue

            h,w   = crop.shape[:2]
            torso = crop[int(h*.25):int(h*.75), int(w*.15):int(w*.85)]
            reg   = torso if torso.size > 0 else crop

            if color_name and color_name.lower() != "any":
                if not detect_color_in_crop(reg, color_name): continue
                label = f"person in {color_name} clothes"
            else:
                label = f"person in {get_dominant_color_name(reg)} clothes"

            key = f"clothes_{x1//50}_{y1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            results_list.append({
                "object":     label,
                "color":      color_name,
                "image_path": save_detection(
                    frame,[x1,y1,x2,y2],label,frame_id,
                    "clothes",(255,165,0)),
                "bbox":      [x1,y1,x2,y2],
                "timestamp": frame_id
            })


# ---- 5. PERSON + VEHICLE PROXIMITY ----

def run_person_vehicle_mode(cap, vehicle_word, vehicle_info, results_list):
    tracker  = CooldownTracker()
    frame_id = 0
    world_model.set_classes(["person"])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        pr = world_model(frame, conf=0.35, imgsz=640)
        if not pr[0].boxes: continue

        p_boxes = pr[0].boxes.xyxy.cpu().numpy()
        v_boxes = detect_vehicles_in_frame(frame, vehicle_info)
        if not v_boxes: continue

        for pbox in p_boxes:
            px1,py1,px2,py2 = map(int, pbox)
            if (px2-px1)*(py2-py1) < AREA_THRESHOLD: continue
            crop = frame[py1:py2, px1:px2]
            if crop.size == 0 or is_blurry(crop): continue

            prox = max(px2-px1, py2-py1) * 2.0
            for vbox in v_boxes:
                vx1,vy1,vx2,vy2 = vbox
                dist = np.sqrt(
                    ((px1+px2)/2-(vx1+vx2)/2)**2 +
                    ((py1+py2)/2-(vy1+vy2)/2)**2
                )
                if dist > prox: continue

                key = f"pv_{px1//50}_{py1//50}"
                if tracker.should_skip(key, frame_id): continue
                tracker.update(key, frame_id)

                ann = frame.copy()
                draw_box(ann, px1,py1,px2,py2, "person",     (0,255,0))
                draw_box(ann, vx1,vy1,vx2,vy2, vehicle_word, (255,100,0))
                path = os.path.join(SAVE_DIR, f"pv_{frame_id}.jpg")
                cv2.imwrite(path, ann, [cv2.IMWRITE_JPEG_QUALITY, 80])

                results_list.append({
                    "object":       f"person with {vehicle_word}",
                    "vehicle":      vehicle_word,
                    "person_bbox":  [px1,py1,px2,py2],
                    "vehicle_bbox": [vx1,vy1,vx2,vy2],
                    "image_path":   path,
                    "timestamp":    frame_id
                })
                break








# ---- 6. GENERIC YOLOWorld FALLBACK ----

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

        res = world_model(frame, conf=0.35, imgsz=640)
        if not res[0].boxes: continue

        boxes = res[0].boxes.xyxy.cpu().numpy()
        clss  = res[0].boxes.cls.cpu().numpy()
        confs = res[0].boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, clss, confs):
            x1,y1,x2,y2 = map(int, box)
            label = prompts[int(cls_id)]
            if (x2-x1)*(y2-y1) < AREA_THRESHOLD: continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or is_blurry(crop): continue

            if ref_feat is not None:
                feat = extract_features(crop)
                sim  = np.dot(ref_feat, feat) / (
                    np.linalg.norm(ref_feat)*np.linalg.norm(feat)+1e-8)
                if sim < 0.65: continue
                conf = float(sim)

            key = f"{label}_{x1//50}_{y1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

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
# MAIN API - UPDATED
# ================================

@app.post("/process")
async def process_video(req: Request):
    data = await req.json()

    video_url = data.get("fileUrl")
    prompt    = data.get("prompt", "person")
    image_url = data.get("imageUrl")

    print(f"\n{'='*60}")
    print(f"🔍 Searching: {prompt}")
    print(f"{'='*60}")

    r = requests.get(video_url, stream=True)
    with open("temp_video.mp4", "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)

    cap = cv2.VideoCapture("temp_video.mp4")
    if not cap.isOpened():
        return {"error": "video open failed"}

    results_list = []

    ref_feat = None
    if image_url:
        ri = requests.get(image_url, stream=True)
        with open("temp_image.jpg", "wb") as f:
            for chunk in ri.iter_content(1024):
                f.write(chunk)
        img = cv2.imread("temp_image.jpg")
        if img is not None:
            ref_feat = extract_features(img)

    parsed = parse_prompt(prompt)
    mode   = parsed["mode"]
    print(f"✅ Mode: {mode}")
    print(f"📋 Parsed: {parsed}")

    if mode == "plate":
        run_plate_mode(cap, parsed["plate_info"], results_list)
    elif mode == "color_object":
        run_color_object_mode(
            cap, parsed["color"],
            parsed["vehicle_word"], parsed["vehicle_info"],
            results_list)
    elif mode == "color_attribute":
        run_color_attribute_mode(
            cap, parsed["object"], parsed["color"], results_list)
    elif mode == "person_attribute":
        run_person_attribute_mode(
            cap, parsed["attributes"], results_list)
    elif mode == "person_attribute_color":  # ← NEW!
        run_person_attribute_color_mode(
            cap, parsed["attribute"], parsed["color"], results_list)
    elif mode == "person_vehicle_color":  # ← NEW!
        run_person_vehicle_color_mode(
            cap, parsed["vehicle_word"], parsed["vehicle_info"],
            parsed["color"], results_list)
    elif mode == "person_vehicle":
        run_person_vehicle_mode(
            cap, parsed["vehicle_word"], parsed["vehicle_info"],
            results_list)
        

    else:
        run_yoloworld_mode(cap, prompt, ref_feat, results_list)

    cap.release()

    print(f"\n📊 Results found: {len(results_list)}")
    print(f"{'='*60}\n")

    return {
        "results":       results_list,
        "mode":          mode,
        "parsed_prompt": parsed,
        "total_found":   len(results_list)
    }