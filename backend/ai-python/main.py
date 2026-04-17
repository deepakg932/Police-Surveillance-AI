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
from deepface import DeepFace
from fastapi.staticfiles import StaticFiles
import re


app = FastAPI()

# --------------------------------
# DEVICE
# --------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)



# --------------------------------
# MODELS
# --------------------------------
# Per-color minimum body-region ratio required for a vehicle match
VEHICLE_COLOR_THRESHOLDS = {
    "red":    0.20,
    "yellow": 0.25,   # yellow is very bright — needs strong presence
    "blue":   0.20,
    "navy":   0.18,
    "green":  0.20,
    "white":  0.24,
    "black":  0.24,
    "orange": 0.22,
    "purple": 0.18,
    "pink":   0.18,
    "silver": 0.22,
    "grey":   0.22,
    "gray":   0.22,
    "brown":  0.18,
    "olive":  0.18,
}

# Colors that are easily confused — target must beat these rivals by 1.5×
VEHICLE_COLOR_CONFUSION = {
    "yellow": ["green", "orange", "white"],
    "orange": ["red", "yellow", "brown"],
    "red":    ["orange", "pink", "brown"],
    "blue":   ["purple", "navy", "black"],
    "green":  ["olive", "yellow", "teal"],
    "purple": ["blue", "pink"],
    "pink":   ["red", "purple", "white"],
    "grey":   ["white", "silver", "black"],
    "gray":   ["white", "silver", "black"],
    "silver": ["white", "grey"],
}

# Prompt-facing color aliases -> canonical HSV color keys
COLOR_SYNONYMS = {
    "maroon": "red",
    "burgundy": "red",
    "crimson": "red",
    "cyan": "blue",
    "teal": "green",
    "lime": "green",
    "beige": "white",
    "cream": "white",
    "off white": "white",
    "off-white": "white",
    "charcoal": "black",
    "magenta": "pink",
    "violet": "purple",
}

# One-time locked calibration profile for vehicle colors.
COLOR_CALIBRATION_PROFILE = {
    "default": {"min_ratio": 0.10, "dominance_mult": 0.90, "relaxed_mult": 0.62, "single_zone_mult": 1.35},
    "red":    {"min_ratio": 0.09, "dominance_mult": 0.72, "relaxed_mult": 0.56, "single_zone_mult": 1.28},
    "blue":   {"min_ratio": 0.10, "dominance_mult": 0.78, "relaxed_mult": 0.60, "single_zone_mult": 1.30},
    "green":  {"min_ratio": 0.10, "dominance_mult": 0.80, "relaxed_mult": 0.60, "single_zone_mult": 1.30},
    "yellow": {"min_ratio": 0.12, "dominance_mult": 0.90, "relaxed_mult": 0.64, "single_zone_mult": 1.35},
    "orange": {"min_ratio": 0.11, "dominance_mult": 0.86, "relaxed_mult": 0.62, "single_zone_mult": 1.32},
    "purple": {"min_ratio": 0.10, "dominance_mult": 0.82, "relaxed_mult": 0.60, "single_zone_mult": 1.30},
    "pink":   {"min_ratio": 0.10, "dominance_mult": 0.82, "relaxed_mult": 0.60, "single_zone_mult": 1.30},
    "black":  {"min_ratio": 0.16, "dominance_mult": 1.05, "relaxed_mult": 0.00, "single_zone_mult": 1.40},
    "white":  {"min_ratio": 0.14, "dominance_mult": 1.05, "relaxed_mult": 0.00, "single_zone_mult": 1.40},
    "grey":   {"min_ratio": 0.12, "dominance_mult": 0.95, "relaxed_mult": 0.00, "single_zone_mult": 1.38},
    "gray":   {"min_ratio": 0.12, "dominance_mult": 0.95, "relaxed_mult": 0.00, "single_zone_mult": 1.38},
    "silver": {"min_ratio": 0.12, "dominance_mult": 0.95, "relaxed_mult": 0.00, "single_zone_mult": 1.38},
}



car_model   = YOLO("yolov8n.pt").to(device)
plate_model = YOLO("license_plate_detector.pt").to(device)
world_model = YOLOWorld("yolov8s-worldv2.pt").to(device)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

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

app.mount("/files", StaticFiles(directory=SAVE_DIR), name="files")

SESSION_ID = None

def get_session_id():
    global SESSION_ID
    if SESSION_ID is None:
        import time
        SESSION_ID = str(int(time.time() * 1000))
    return SESSION_ID

FRAME_SKIP     = 5  # check more frames for plates
AREA_THRESHOLD = 300
BLUR_THRESHOLD = 15
COOLDOWN       = 5
STRICT_PROMPT_MATCH = True

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])





CLOTHING_YOLO_LABELS = {
    "shirt":       ["shirt", "dress shirt", "casual shirt", "collared shirt", "formal shirt"],
    "tshirt":      ["t-shirt", "tshirt", "tee shirt", "polo shirt", "casual top"],
    "t-shirt":     ["t-shirt", "tshirt", "tee shirt", "polo shirt", "casual top"],
    "jacket":      ["jacket", "zip-up jacket", "bomber jacket", "denim jacket"],
    "coat":        ["coat", "overcoat", "winter coat", "long coat", "trench coat"],
    "jeans":       ["jeans", "denim pants", "blue jeans"],
    "pant":        ["pants", "trousers", "slacks", "track pants"],
    "dress":       ["dress", "frock", "gown"],
    "sweater":     ["sweater", "pullover", "knit top"],
    "hoodie":      ["hoodie", "sweatshirt", "zip hoodie", "hooded jacket"],
    "kurta":       ["kurta", "long shirt", "tunic"],
    "saree":       ["saree", "sari", "indian saree", "traditional saree"],
    "salwar suit": ["salwar suit", "salwar kameez", "churidar suit", "kurta pyjama", "ethnic wear"],
    "kurti":       ["kurti", "short kurta", "tunic top"],
    "blazer":      ["blazer", "sport coat", "suit jacket"],
    "suit":        ["suit", "business suit", "formal suit"],
    "top":         ["top", "blouse", "shirt"],
    "lehenga":     ["lehenga", "skirt", "ethnic skirt"],
}

COLOR_THRESHOLDS = {
    "yellow": 0.18,   # Yellow needs strong presence — avoids false positives
    "red":    0.12,
    "blue":   0.12,
    "navy":   0.10,
    "green":  0.12,
    "orange": 0.15,
    "purple": 0.12,
    "pink":   0.12,
    "black":  0.10,
    "white":  0.12,
    "grey":   0.10,
    "gray":   0.10,
    "silver": 0.10,
    "brown":  0.10,
    "olive":  0.10,
}



def get_vehicle_body_crop(frame, box, padding_ratio=0.05):
    """
    Extract the vehicle body region, stripping road, sky and background.
    
    Strategy:
    - Horizontal: use center 80% (strip door-frame edges that may be background)
    - Vertical: use 15%–80% of height (strip roof antenna / road shadow)
    - Extra: reject if crop mean brightness is too close to pure white (sky) 
      or pure black (shadow on road)
    """
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    w = x2 - x1

    # Add small padding so we don't clip tight bounding box edges
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame.shape[1], x2 + pad_x)
    y2 = min(frame.shape[0], y2 + pad_y)

    full = frame[y1:y2, x1:x2]
    if full.size == 0:
        return None, None, None

    h2, w2 = full.shape[:2]

    # Three zones for multi-zone voting
    zone_a = full[int(h2 * 0.05):int(h2 * 0.38), int(w2 * 0.10):int(w2 * 0.90)]  # roof/hood
    zone_b = full[int(h2 * 0.35):int(h2 * 0.72), int(w2 * 0.08):int(w2 * 0.92)]  # doors
    zone_c = full[int(h2 * 0.68):int(h2 * 0.88), int(w2 * 0.15):int(w2 * 0.85)]  # lower panels

    return zone_a, zone_b, zone_c


def vehicle_color_match(frame, box, color_name):
    """
    Multi-zone color validation for vehicles.
    
    Returns (matched: bool, confidence: float, details: dict)
    
    Requires at least 2 of 3 zones to exceed threshold.
    Also applies confusion-pair exclusion.
    """
    if color_name is None:
        return True, 1.0, {}   # any-color query — always pass
    color_name = normalize_color_name(color_name)
    calib = get_color_calibration(color_name)

    zone_a, zone_b, zone_c = get_vehicle_body_crop(frame, box)
    zones = [(z, name) for z, name in [(zone_a, "roof"), (zone_b, "doors"), (zone_c, "lower")]
             if z is not None and z.size > 0]

    if not zones:
        return False, 0.0, {}

    threshold = VEHICLE_COLOR_THRESHOLDS.get(color_name.lower(), 0.20)
    confusion_rivals = VEHICLE_COLOR_CONFUSION.get(color_name.lower(), [])
    neutral_colors = {"black", "white", "grey", "gray", "silver"}

    zone_results = []
    for crop, zone_name in zones:
        target_r = color_ratio(crop, color_name)

        # Must exceed base threshold
        if target_r < threshold:
            zone_results.append((False, target_r, zone_name))
            continue

        # Confusion-pair check: target must be ≥1.5× each rival
        confused = False
        for rival in confusion_rivals:
            rival_r = color_ratio(crop, rival)
            if rival_r > 0 and target_r < rival_r * 1.5:
                confused = True
                break
        if confused:
            zone_results.append((False, target_r, zone_name))
            continue

        # Dominance check: target must be the most present color.
        # Neutral colors need stricter separation (black/white/grey/silver are commonly confused).
        max_other = max(
            (color_ratio(crop, c) for c in COLOR_RANGES if c != color_name),
            default=0.0
        )
        if color_name in neutral_colors:
            if max_other > target_r * 1.05:
                zone_results.append((False, target_r, zone_name))
                continue
        else:
            dominance_limit = max(1.05, 1.0 / max(0.55, calib.get("dominance_mult", 0.90)))
            if max_other > target_r * dominance_limit:
                zone_results.append((False, target_r, zone_name))
                continue

        zone_results.append((True, target_r, zone_name))

    passed = [r for r in zone_results if r[0]]
    failed = [r for r in zone_results if not r[0]]

    # Adaptive acceptance:
    # - normal: at least 2 zones pass
    # - fallback: if 1 zone is very strong, accept (helps short/angled vehicle clips)
    strong_single = any(r[1] >= threshold * calib.get("single_zone_mult", 1.35) for r in passed)
    if len(passed) < 2 and not (len(passed) == 1 and strong_single):
        return False, 0.0, {"zones": zone_results}

    avg_conf = sum(r[1] for r in passed) / len(passed)
    return True, round(avg_conf, 3), {
        "zones": zone_results,
        "passed": len(passed),
        "fallback_single_zone": (len(passed) == 1 and strong_single)
    }


class VehicleTracker:
    """
    Track vehicle detections across frames.
    Only emit a final detection once 3 out of 5 frames agree on the color.
    Prevents single-frame false positives from lighting/shadow.
    """
    def __init__(self, required_votes=3, window=5, iou_threshold=0.40):
        self._tracks = {}   # key → {"votes": int, "misses": int, "last_frame": int, "box": list}
        self._required = required_votes
        self._window = window
        self._iou_thresh = iou_threshold

    def _find_track(self, box):
        for key, track in self._tracks.items():
            if iou(box, track["box"]) >= self._iou_thresh:
                return key
        return None

    def update(self, box, frame_id, matched):
        """
        matched=True  → this frame found the color on this vehicle
        matched=False → this vehicle was detected but color didn't match
        Returns True if accumulated votes hit the required threshold.
        """
        key = self._find_track(box)
        if key is None:
            key = f"{int(box[0])//40}_{int(box[1])//40}_{frame_id}"
            self._tracks[key] = {"votes": 0, "misses": 0, "last_frame": frame_id, "box": list(box)}

        track = self._tracks[key]
        track["last_frame"] = frame_id
        track["box"] = list(box)   # update position

        if matched:
            track["votes"] += 1
            return track["votes"] >= self._required
        else:
            track["misses"] += 1
            return False

    def cleanup(self, frame_id, max_age=30):
        dead = [k for k, v in self._tracks.items()
                if frame_id - v["last_frame"] > max_age]
        for k in dead:
            del self._tracks[k]


def get_color_threshold(color_name):
    return COLOR_THRESHOLDS.get(color_name.lower(), 0.12)

def get_clothing_torso_region(person_crop, clothing_type):
    """
    Clothing type ke hisab se correct body region return karo.
    Shirt/tshirt = upper body, Jeans/pant = lower body, Dress = full
    """
    h, w = person_crop.shape[:2]
    clothing_type = clothing_type.lower()

    UPPER_BODY = {"shirt", "tshirt", "t-shirt", "jacket", "coat",
                  "sweater", "hoodie", "kurta", "kurti", "blazer",
                  "suit", "top", "salwar suit"}
    LOWER_BODY = {"jeans", "pant", "lehenga", "trousers"}
    FULL_BODY  = {"dress", "saree", "gown"}

    if clothing_type in UPPER_BODY:
        # Upper body: 15% to 60% of person height
        return person_crop[int(h * 0.15):int(h * 0.60),
                           int(w * 0.10):int(w * 0.90)]
    elif clothing_type in LOWER_BODY:
        # Lower body: 50% to 90%
        return person_crop[int(h * 0.50):int(h * 0.90),
                           int(w * 0.10):int(w * 0.90)]
    else:
        # Full body (dress, saree): 10% to 90%
        return person_crop[int(h * 0.10):int(h * 0.90),
                           int(w * 0.10):int(w * 0.90)]

def is_color_dominant(crop, color_name, threshold=None):
    """
    Color check with dominance validation.
    Target color must be present AND must beat competitor colors.
    """
    if crop is None or crop.size == 0:
        return False, 0.0

    if threshold is None:
        threshold = get_color_threshold(color_name)

    target_ratio = color_ratio(crop, color_name)

    if target_ratio < threshold:
        return False, target_ratio

    # Dominance check: target color ka ratio competitor se significantly zyada hona chahiye
    max_competitor = 0.0
    for c in COLOR_RANGES:
        if c == color_name:
            continue
        r = color_ratio(crop, c)
        if r > max_competitor:
            max_competitor = r

    # Special rules for tricky colors
    color_lower = color_name.lower()

    if color_lower in ("black", "white", "grey", "gray", "silver"):
        # Neutral colors: competitor 1.5x se zyada nahi hona chahiye
        if max_competitor > target_ratio * 1.5:
            return False, target_ratio
    else:
        # Vivid colors: target must be clearly dominant
        if max_competitor > target_ratio * 1.2:
            return False, target_ratio

    return True, target_ratio



def person_clothing_spatial_check(p_box, c_box, clothing_type):
    px1, py1, px2, py2 = p_box
    cx1, cy1, cx2, cy2 = c_box

    ph = py2 - py1
    pw = px2 - px1
    ch = cy2 - cy1
    cw = cx2 - cx1

    if ph <= 0 or pw <= 0 or ch <= 0 or cw <= 0:
        return False

    # 1. Clothing center person box ke andar hona chahiye (strict)
    ccx = (cx1 + cx2) / 2
    ccy = (cy1 + cy2) / 2
    if not (px1 - pw * 0.05 <= ccx <= px2 + pw * 0.05):
        return False

    # 2. Clothing type ke hisab se vertical position
    clothing_lower = clothing_type.lower()
    UPPER = {"shirt","tshirt","t-shirt","jacket","coat",
              "sweater","hoodie","kurta","kurti","blazer","suit","top"}
    LOWER = {"jeans","pant","trousers","lehenga"}

    rel_center = (ccy - py1) / ph

    if clothing_lower in UPPER:
        if not (0.10 <= rel_center <= 0.72):
            return False
    elif clothing_lower in LOWER:
        if not (0.42 <= rel_center <= 0.95):
            return False
    else:  # dress, saree, full body
        if not (0.15 <= rel_center <= 0.85):
            return False

    # 3. Clothing height — relaxed
    if not (0.15 * ph <= ch <= 0.85 * ph):
        return False

    # 4. Clothing width — relaxed
    if not (0.25 * pw <= cw <= 1.05 * pw):
        return False

    # 5. ✅ SAHI OVERLAP CHECK
    # Clothing ka center person ke andar hai — overlap bhi hona chahiye
    overlap_x = max(0, min(px2, cx2) - max(px1, cx1))
    overlap_y = max(0, min(py2, cy2) - max(py1, cy1))
    overlap_area = overlap_x * overlap_y
    person_area  = pw * ph

    # Overlap kam se kam person area ka 8% hona chahiye
    if overlap_area < person_area * 0.08:
        return False

    # 6. ✅ BACKGROUND PERSON REJECTION (tera main problem)
    # Clothing box predominantly person ke andar hona chahiye
    cloth_area = cw * ch
    if cloth_area > 0:
        cloth_inside_ratio = overlap_area / cloth_area
        if cloth_inside_ratio < 0.50:  # 50% clothing person ke andar honi chahiye
            return False

    return True


def ethnic_clothing_guard(p_box, c_box, clothing_type):
    """
    Extra guard for saree/salwar-suit:
    reject upper-body-only garments (tshirt/shirt/jacket) being mislabeled.
    """
    clothing_type = normalize_clothing_type(clothing_type)
    if clothing_type not in {"saree", "salwar suit"}:
        return True

    px1, py1, px2, py2 = p_box
    cx1, cy1, cx2, cy2 = c_box
    ph = max(1, py2 - py1)
    ch = max(1, cy2 - cy1)

    rel_top = (cy1 - py1) / ph
    rel_bottom = (cy2 - py1) / ph
    rel_h = ch / ph

    # Saree/salwar suit should usually span much of torso+lower body.
    if rel_h < 0.45:
        return False
    if rel_bottom < 0.70:
        return False
    if rel_top > 0.42:
        return False
    return True

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
    return re.sub(r'[^A-Z0-9\u4e00-\u9fff]', '', text.upper())





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
    raw = (prompt or "").strip().upper()
    if not raw:
        return {"type": "not_plate"}

    # Only treat as plate query when user clearly asked for plate / numberplate,
    # OR provided a plate-like token (not generic IDs / short numbers).
    if not re.search(r'\b(plate|number\s*plate|license\s*plate|licence\s*plate)\b', raw, flags=re.I):
        # If prompt doesn't mention plate keywords, allow plate parsing only for
        # strong plate-like strings (Indian full plate patterns).
        compact = re.sub(r'[^A-Z0-9]', '', raw)
        parsed_strong = parse_plate_string(compact)
        if not (parsed_strong and parsed_strong.get("state") in INDIA_STATE_CODES):
            return {"type": "not_plate"}
    clean = re.sub(r'[^A-Z0-9]', '', raw)
    parts = raw.split()

    # ---- 1. Only state code  e.g. "MP", "UP", "DL"
    if clean in INDIA_STATE_CODES and len(clean) == 2:
        return {
            "type": "state_only",
            "state": clean,
            "desc": f"All vehicles from {INDIA_STATE_CODES[clean]}"
        }

    # ---- 2. State + RTO  e.g. "MP09", "UP32", "MH04"
    m = re.match(r'^([A-Z]{2})\s?(\d{1,2})$', raw)
    if m and m.group(1) in INDIA_STATE_CODES:
        st = m.group(1)
        rt = m.group(2).zfill(2)
        return {
            "type": "state_rto",
            "state": st,
            "rto": rt,
            "prefix": st + rt,
            "desc": f"RTO {st}-{rt} ({INDIA_STATE_CODES[st]})"
        }

    # ---- 3. Full plate (Indian format)
    parsed = parse_plate_string(raw)
    if parsed and parsed["state"] in INDIA_STATE_CODES:
        return {
            "type": "full_plate",
            "parsed": parsed,
            "desc": f"Exact plate {parsed['full']}"
        }

    # ---- 4. State + last digits (half & half)  e.g. "MP 7785"
    if len(parts) == 2:
        st_part = re.sub(r'[^A-Z]', '', parts[0])
        num_part = re.sub(r'[^0-9]', '', parts[1])
        if st_part in INDIA_STATE_CODES and num_part:
            return {
                "type": "state_number",
                "state": st_part,
                "number": num_part,
                "desc": f"State={st_part} AND ends with {num_part}"
            }

    # ---- 5. Half plate: series+number only  e.g. "ZP7785", "7785"
    # Guard: avoid triggering on short numbers/IDs unless plate keywords were present.
    if any(c.isdigit() for c in clean) and clean not in INDIA_STATE_CODES:
        if len(clean) < 4:
            return {"type": "not_plate"}
        return {
            "type": "half_plate",
            "partial": clean,
            "desc": f"Plate contains '{clean}'"
        }

    # ---- 6. International / generic plate (fallback)
    cleaned = re.sub(r'\s+', '', raw)
    if len(cleaned) >= 4 and not cleaned.isdigit():
        return {
            "type": "full_plate",
            "parsed": {"full": cleaned},
            "desc": f"International plate '{cleaned}'"
        }

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
    ptype = plate_info["type"]
    plate_vars = ocr_variants(plate)

    if ptype == "state_only":
        state = plate_info["state"]
        # Exact
        if any(v.startswith(state) for v in plate_vars):
            return True
        # Fuzzy on first 2 chars
        if len(plate) >= 2:
            if SequenceMatcher(None, state, plate[:2]).ratio() >= 0.8:
                return True
        return False

    elif ptype == "state_rto":
        prefix = plate_info["prefix"]
        if any(v.startswith(prefix) for v in plate_vars):
            return True
        if len(plate) >= 4:
            if SequenceMatcher(None, prefix, plate[:4]).ratio() >= 0.78:
                return True
        return False

    elif ptype == "full_plate":
        target = plate_info["parsed"]["full"]
        target_vars = ocr_variants(target)
        # Exact
        if plate in target_vars or any(tv in plate_vars for tv in target_vars):
            return True
        # Fuzzy ratio
        if SequenceMatcher(None, target, plate).ratio() >= 0.72:
            return True
        return False

    elif ptype == "state_number":
        state, number = plate_info["state"], plate_info["number"]
        state_ok = any(v.startswith(state) for v in plate_vars)
        if not state_ok and len(plate) >= 2:
            state_ok = SequenceMatcher(None, state, plate[:2]).ratio() >= 0.8
        num_ok = any(number in v for v in plate_vars)
        if not num_ok:
            tail = plate[-len(number):]
            num_ok = SequenceMatcher(None, number, tail).ratio() >= 0.8
        return state_ok and num_ok

    elif ptype == "half_plate":
        partial = plate_info["partial"]
        partial_vars = ocr_variants(partial)
        if any(pv in plate for pv in partial_vars):
            return True
        if SequenceMatcher(None, partial, plate).ratio() >= 0.75:
            return True
        # sliding window
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



    # Version 3: CLAHE + Gaussian
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    v3    = clahe.apply(gray)
    v3    = cv2.GaussianBlur(v3, (3, 3), 0)
    versions.append(v3)



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
        results = reader.readtext(enhanced, detail=0)
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

COLOR_CONFUSION_PAIRS = {
    "yellow": ["orange", "green", "white"],   # yellow often confused with these
    "orange": ["red", "yellow", "brown"],
    "red":    ["orange", "pink", "brown"],
    "blue":   ["purple", "navy", "black"],
    "green":  ["olive", "yellow", "teal"],
    "purple": ["blue", "pink"],
    "pink":   ["red", "purple", "white"],
}

TWO_WHEELER_INFO = {"strategy": "world", "world_label": "motorcycle"}   # ya "bike", "scooter"

COLOR_RANGES = {
    "red":    [([0,   80,  60], [10,  255, 255]),
               ([170, 80,  60], [180, 255, 255])],
    "yellow": [([20,  100, 100], [35,  255, 255])],   # TIGHT — high sat required
    "blue":   [([90,  60,  40], [130, 255, 255])],    # Clearly separated from yellow
    "navy":   [([90,  60,  20], [130, 255, 120])],    # Dark blue
    "green":  [([36,  50,  40], [85,  255, 255])],
    "white":  [([0,   0,  180], [180,  30, 255])],
    "black":  [([0,   0,    0], [180, 255,  55])],
    "orange": [([8,   120, 80], [22,  255, 255])],
    "purple": [([120, 50,  40], [160, 255, 255])],
    "pink":   [([140, 40,  80], [175, 255, 255])],
    "silver": [([0,   0,  130], [180,  35, 210])],
    "grey":   [([0,   0,   50], [180,  28, 165])],
    "gray":   [([0,   0,   50], [180,  28, 165])],
    "brown":  [([8,   40,  30], [22,  200, 170])],
    "olive":  [([25,  30,  30], [65,  255, 155])],
    "olive green": [([25, 30, 30], [65, 255, 155])],
}
 
# Per-color minimum ratio — how much of the crop must be this color




# Shoe types with descriptive names for YOLOWorld
SHOE_TYPE_MAP = {
    'shoes': 'shoe (footwear)',
    'shoe': 'shoe (footwear)',
    'sneakers': 'sneakers (footwear)',
    'sandals': 'sandals (footwear)',
    'boots': 'boots (footwear)',
}


def normalize_shoe_type(shoe_type: str):
    if not shoe_type:
        return "shoes"
    s = shoe_type.strip().lower()
    alias = {
        "shoe": "shoes",
        "shoes": "shoes",
        "sneaker": "sneakers",
        "sneakers": "sneakers",
        "boot": "boots",
        "boots": "boots",
        "sandal": "sandals",
        "sandals": "sandals",
        "chappal": "sandals",
        "chappals": "sandals",
        "footwear": "footwear",
    }
    return alias.get(s, s)


def canonicalize_detected_shoe_label(label: str):
    s = (label or "").strip().lower()
    if any(k in s for k in ["sandal", "flip", "flop", "slipper", "open toe", "chappal"]):
        return "sandals"
    if "boot" in s:
        return "boots"
    if any(k in s for k in ["sneaker", "athletic", "running", "sports"]):
        return "sneakers"
    if any(k in s for k in ["shoe", "footwear"]):
        return "shoes"
    return "shoes"

FOOTWEAR_CLASSES = {
    "shoes":    ["closed toe shoes", "sneakers", "boots", "sports shoes", "running shoes"],
    "sneakers": ["sneakers", "sports shoes", "running shoes", "athletic shoes"],
    "boots":    ["boots", "ankle boots", "leather boots"],
    "sandals":  ["sandals", "flip flops", "slippers", "open toe shoes"],
    "footwear": ["shoes", "sneakers", "boots", "sandals"],
}

# ================================
# VEHICLE ALIASES
# ================================

VEHICLE_ALIASES = {
    # Four-wheelers (YOLO class 2)
    "car": {"strategy": "yolo_class", "class_ids": [2]},
    "cars": {"strategy": "yolo_class", "class_ids": [2]},
    "vehicle":      {"strategy": "yolo_class", "class_ids": [2, 3, 5, 7]},
    "vehicles":     {"strategy": "yolo_class", "class_ids": [2, 3, 5, 7]},
    "suv":           {"strategy": "world", "world_label": "SUV"},
    "jeep":          {"strategy": "world", "world_label": "jeep"},
    "pickup":        {"strategy": "world", "world_label": "pickup truck"},
    "van":           {"strategy": "world", "world_label": "van"},
   
    # Trucks (YOLO class 7)
    "truck":         {"strategy": "yolo_class", "class_ids": [7]},
    "trucks":        {"strategy": "yolo_class", "class_ids": [7]},
    "lorry":         {"strategy": "yolo_class", "class_ids": [7]},
   
    # Buses (YOLO class 5)
    "bus":           {"strategy": "yolo_class", "class_ids": [5]},
    "buses":         {"strategy": "yolo_class", "class_ids": [5]},
   
    # Two-wheelers - Use YOLOWorld for accurate distinction!
    "bike":          {"strategy": "world", "world_label": "bike"},
    "bikes":         {"strategy": "world", "world_label": "bike"},
    "motorcycle":    {"strategy": "world", "world_label": "motorcycle"},
    "motorcycles":   {"strategy": "world", "world_label": "motorcycle"},
    "motorbike":     {"strategy": "world", "world_label": "motorbike"},
    "motorbikes":    {"strategy": "world", "world_label": "motorbike"},
    "scooter":       {"strategy": "world", "world_label": "scooter"},
    "scooters":      {"strategy": "world", "world_label": "scooter"},
    "activa":        {"strategy": "world", "world_label": "scooter"},
    "vespa":         {"strategy": "world", "world_label": "scooter"},
    "moped":         {"strategy": "world", "world_label": "scooter"},
   
    # Auto rickshaw / Three wheelers - SEPARATE class!
    "auto":          {"strategy": "world", "world_label": "auto rickshaw"},
    "autos":         {"strategy": "world", "world_label": "auto rickshaw"},
    "rickshaw":      {"strategy": "world", "world_label": "auto rickshaw"},
    "rickshaws":     {"strategy": "world", "world_label": "auto rickshaw"},
    "auto rickshaw": {"strategy": "world", "world_label": "auto rickshaw"},
    "tuk tuk":       {"strategy": "world", "world_label": "auto rickshaw"},
    "tuktuk":        {"strategy": "world", "world_label": "auto rickshaw"},
    "tempo":         {"strategy": "world", "world_label": "auto rickshaw"},
    # Electric rickshaw must stay separate from auto-rickshaw.
    "e-rickshaw":        {"strategy": "world", "world_label": "electric rickshaw"},
    "e rickshaw":        {"strategy": "world", "world_label": "electric rickshaw"},
    "electric rickshaw": {"strategy": "world", "world_label": "electric rickshaw"},
    "electric-rickshaw": {"strategy": "world", "world_label": "electric rickshaw"},
    "erickshaw":         {"strategy": "world", "world_label": "electric rickshaw"},
    # Common Hinglish typos
    "e-rikshaw":         {"strategy": "world", "world_label": "electric rickshaw"},
    "e rikshaw":         {"strategy": "world", "world_label": "electric rickshaw"},
    "e-riksaw":          {"strategy": "world", "world_label": "electric rickshaw"},
    "e riksaw":          {"strategy": "world", "world_label": "electric rickshaw"},
    "e-riksw":           {"strategy": "world", "world_label": "electric rickshaw"},
    "e riksw":           {"strategy": "world", "world_label": "electric rickshaw"},
    "e-riksahw":         {"strategy": "world", "world_label": "electric rickshaw"},
    "e riksahw":         {"strategy": "world", "world_label": "electric rickshaw"},
   
    # Other vehicles
    "ambulance":     {"strategy": "world", "world_label": "ambulance"},
    "taxi":          {"strategy": "world", "world_label": "taxi"},
    "cab":           {"strategy": "world", "world_label": "taxi"},
}

PERSON_ATTRIBUTE_KEYWORDS = [
    "helmet", "bag", "weapon", "gun", "knife", "cap", "hat",
    "backpack", "jacket", "coat", "shirt", "tshirt", "t-shirt", "shirt",  "mask", "uniform", "vest",
    "bag","backpack","handbag","schoolbag","school bag",
    "gloves","shoes","weapon","gun",
    "glasses", "sunglasses", "gloves", "shoes","backpack", "handbag", "school bag"
]

HEAD_ATTRS   = {"helmet", "cap", "hat", "mask", "glasses", "sunglasses"}
BODY_ATTRS   = {"bag", "backpack", "jacket", "vest", "uniform", "gloves", "shoes"}
WEAPON_ATTRS = {"weapon", "gun", "knife"}


def is_color_match_for_bag(crop, color_name):
    """
    Strict color match specifically for bag detection.
    Returns (matched: bool, ratio: float, dominant_color: str)
    """
    if crop is None or crop.size == 0:
        return False, 0.0, "unknown"
 
    color_name = normalize_color_name(color_name)
    neutral = {"black", "white", "grey", "gray", "silver"}
    if color_name in neutral:
        threshold = max(0.07, get_color_threshold(color_name) * 0.70)
    else:
        threshold = max(0.09, get_color_threshold(color_name) * 0.90)
    target_ratio = color_ratio(crop, color_name)
 
    if target_ratio < threshold:
        return False, target_ratio, get_dominant_color_name(crop)
 
    # Find the actual dominant color
    best_color = color_name
    best_ratio = target_ratio
    for c in COLOR_RANGES:
        if c == color_name:
            continue
        r = color_ratio(crop, c)
        if r > best_ratio:
            best_color = c
            best_ratio = r
 
    # Extra allowance for white/neutral bags under sunlight/reflection.
    if color_name == "white":
        max_comp = 0.0
        for c in COLOR_RANGES:
            if c == "white":
                continue
            r = color_ratio(crop, c)
            if r > max_comp:
                max_comp = r
        if target_ratio >= threshold and max_comp <= target_ratio * 1.25:
            return True, target_ratio, "white"

    # Target should be dominant, but allow near-dominance for vivid colors.
    if best_color != color_name:
        if color_name in neutral:
            return False, target_ratio, best_color
        if best_ratio > target_ratio * 1.20:
            return False, target_ratio, best_color
 
    # Extra strictness for confusion-prone colors
    confused_with = COLOR_CONFUSION_PAIRS.get(color_name.lower(), [])
    for confused_color in confused_with:
        confused_ratio = color_ratio(crop, confused_color)
        if color_name in neutral:
            if confused_ratio > target_ratio * 0.80:
                return False, target_ratio, confused_color
        else:
            if confused_ratio > target_ratio * 0.92:
                return False, target_ratio, confused_color
 
    return True, target_ratio, color_name


def is_valid_person_box(box, frame_shape, min_area=500, max_frame_ratio=0.75):
    """
    Returns True only if person box is reasonable size.
    Rejects boxes that are wider or taller than 75% of frame.
    """
    x1, y1, x2, y2 = box
    frame_h, frame_w = frame_shape[:2]
    pw = x2 - x1
    ph = y2 - y1
    area = pw * ph
 
    if area < min_area:
        return False
 
    # Reject if box is unreasonably large (close-up person from behind)
    # Still allow large persons — just not >75% of frame in both dims
    if pw > frame_w * max_frame_ratio and ph > frame_h * max_frame_ratio:
        return False
 
    # Aspect ratio: person should be taller than wide (0.3 < w/h < 1.2)
    if ph > 0:
        aspect = pw / ph
        if aspect > 1.5:  # wider than 1.5x height = not a standing person
            return False
 
    return True



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
    session = get_session_id()
    filename = f"{prefix}_{session}_{frame_id}.jpg"

    full_path = os.path.join(SAVE_DIR, filename)
    draw_box(ann, x1, y1, x2, y2, label, color)
   
    cv2.imwrite(full_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return filename


def color_ratio(crop, color_name):
    if crop is None or crop.size == 0:
        return 0.0
    color_name = normalize_color_name(color_name)
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES.get(color_name.lower(), []):
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv, np.array(lo, dtype=np.uint8),
                              np.array(hi, dtype=np.uint8))
        )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1] + 1e-5)


def detect_color_in_crop(crop, color_name, threshold=0.25):
    return color_ratio(crop, color_name) > threshold


def get_dominant_color_name(crop):
    """Return the name of the most dominant color in crop."""
    if crop is None or crop.size == 0:
        return "unknown"
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
        alias_pattern = r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b"
        if re.search(alias_pattern, p):
            return alias, VEHICLE_ALIASES[alias]
    return None, None


def extract_color_from_prompt(p: str):
    """Return best color keyword from prompt using word-boundary match."""
    available = list(COLOR_RANGES.keys()) + list(COLOR_SYNONYMS.keys())
    for color_name in sorted(available, key=len, reverse=True):
        color_pattern = r"\b" + re.escape(color_name).replace(r"\ ", r"\s+") + r"\b"
        if re.search(color_pattern, p):
            return normalize_color_name(color_name)
    return None


def normalize_color_name(color_name: str):
    """Normalize prompt color to canonical COLOR_RANGES key."""
    if color_name is None:
        return None
    c = color_name.strip().lower()
    if c in COLOR_RANGES:
        return c
    return COLOR_SYNONYMS.get(c, c)


def normalize_clothing_type(clothing_type: str):
    if not clothing_type:
        return ""
    c = clothing_type.strip().lower().replace("-", "").replace(" ", "")
    mapping = {
        "tshirt": "tshirt",
        "shirt": "shirt",
        "jacket": "jacket",
        "coat": "coat",
        "saree": "saree",
        "sareee": "saree",
        "sareeee": "saree",
        "sari": "saree",
        "sare": "saree",
        "salwarsuit": "salwar suit",
        "salwarsit": "salwar suit",
        "salwarsut": "salwar suit",
        "salwarsiut": "salwar suit",
        "salwarsuitt": "salwar suit",
        "salwarsuite": "salwar suit",
        "salwarsuut": "salwar suit",
        "salwarsot": "salwar suit",
        "salwarshit": "salwar suit",
        "salwarset": "salwar suit",
        "salwarsiit": "salwar suit",
        "salwarsittt": "salwar suit",
        "salwarsuitt": "salwar suit",
        "salwarsiittt": "salwar suit",
        "salwarsiitt": "salwar suit",
        "salwarsitit": "salwar suit",
        "salwarsite": "salwar suit",
        "salwarsit ": "salwar suit",
        "salwarsitx": "salwar suit",
        "salwarsitz": "salwar suit",
        "salwarsitq": "salwar suit",
        "salwarsitw": "salwar suit",
        "salwarsita": "salwar suit",
        "salwarsitb": "salwar suit",
        "salwarsitc": "salwar suit",
        "salwar suit": "salwar suit",
        "salwarsuits": "salwar suit",
        "salwar kameez": "salwar suit",
        "kurti": "kurti",
        "kurta": "kurta",
        "dress": "dress",
        "hoodie": "hoodie",
        "hoddie": "hoodie",
        "huddy": "hoodie",
        "hoody": "hoodie",
        "sweater": "sweater",
        "blazer": "blazer",
        "suit": "suit",
        "jeans": "jeans",
        "pant": "pant",
        "trousers": "trousers",
        "top": "top",
        "uniform": "uniform",
    }
    return mapping.get(c, clothing_type.strip().lower())


def ethnic_color_region_guard(person_crop, color_name, clothing_type):
    """
    Prevent tshirt/shirt from being mislabeled as saree/salwar suit in color mode.
    Requires meaningful target color in lower-body region.
    """
    clothing_type = normalize_clothing_type(clothing_type)
    color_name = normalize_color_name(color_name)
    if person_crop is None or person_crop.size == 0:
        return False
    if clothing_type not in {"saree", "salwar suit"}:
        return True

    h, w = person_crop.shape[:2]
    upper = person_crop[int(h * 0.18):int(h * 0.52), int(w * 0.12):int(w * 0.88)]
    lower = person_crop[int(h * 0.52):int(h * 0.94), int(w * 0.12):int(w * 0.88)]
    if upper.size == 0 or lower.size == 0:
        return False

    upper_r = color_ratio(upper, color_name)
    lower_r = color_ratio(lower, color_name)

    if clothing_type == "saree":
        return lower_r >= 0.08 and (upper_r + lower_r) >= 0.16
    # salwar suit usually has both upper + lower contribution
    return upper_r >= 0.06 and lower_r >= 0.05


def get_color_calibration(color_name: str):
    c = normalize_color_name(color_name) if color_name else "default"
    return COLOR_CALIBRATION_PROFILE.get(c, COLOR_CALIBRATION_PROFILE["default"])


def _extract_normalized_color_from_text(text: str):
    if not text:
        return None
    return extract_color_from_prompt(str(text).lower())


def enforce_prompt_exactness(parsed: dict, results: list):
    """
    Global strict post-filter.
    Keeps only results that satisfy the parsed prompt intent.
    """
    if not STRICT_PROMPT_MATCH or not isinstance(parsed, dict):
        return results

    mode = parsed.get("mode")
    filtered = []

    req_color = normalize_color_name(parsed.get("color")) if parsed.get("color") else None
    req_vehicle = str(parsed.get("vehicle_word", "")).lower().strip()
    req_gender = str(parsed.get("gender", "")).lower().strip()
    req_bag = str(parsed.get("bag_type", "")).lower().strip()
    req_shoe = str(parsed.get("shoe_type", "")).lower().strip()
    req_clothing = normalize_clothing_type(str(parsed.get("clothing_type", "")).lower().strip())

    for r in results:
        obj = str(r.get("object", "")).lower()
        r_color = normalize_color_name(r.get("color")) if r.get("color") else None
        text_color = _extract_normalized_color_from_text(obj)
        color_seen = r_color or text_color

        if mode == "color_object":
            if req_color and color_seen != req_color:
                continue
            if req_vehicle and req_vehicle not in obj and str(r.get("vehicle", "")).lower() != req_vehicle:
                continue
            filtered.append(r)
            continue

        if mode == "person_vehicle_color":
            if req_color and color_seen != req_color:
                continue
            if req_vehicle and req_vehicle not in obj and str(r.get("vehicle", "")).lower() != req_vehicle:
                continue
            filtered.append(r)
            continue

        if mode == "person_vehicle":
            if req_vehicle and req_vehicle not in obj and str(r.get("vehicle", "")).lower() != req_vehicle:
                continue
            filtered.append(r)
            continue

        if mode == "person_shoes_any":
            if req_shoe:
                r_shoe = normalize_shoe_type(str(r.get("shoe_type", "")).lower())
                if req_shoe != "footwear" and r_shoe != req_shoe and req_shoe not in obj:
                    continue
            filtered.append(r)
            continue

        if mode in {"person_color_only", "gender_color", "person_clothing_color",
                    "person_bag_color", "person_shoes_color", "person_helmet_color"}:
            if req_color and color_seen != req_color:
                continue
            if mode == "gender_color" and req_gender and req_gender not in obj and str(r.get("gender", "")).lower() != req_gender:
                continue
            if mode == "person_bag_color" and req_bag and req_bag not in obj and str(r.get("bag_type", "")).lower() != req_bag:
                continue
            if mode == "person_shoes_color" and req_shoe and req_shoe not in obj and str(r.get("shoe_type", "")).lower() != req_shoe:
                continue
            if mode == "person_clothing_color" and req_clothing and req_clothing not in obj and normalize_clothing_type(str(r.get("clothing_type", "")).lower()) != req_clothing:
                continue
            filtered.append(r)
            continue

        if mode == "person_clothing_any":
            if req_clothing and req_clothing not in obj and normalize_clothing_type(str(r.get("clothing_type", "")).lower()) != req_clothing:
                continue
            filtered.append(r)
            continue

        # For other modes, keep original behavior.
        filtered.append(r)

    return filtered


def run_person_color_only_mode(cap, color_name, gender, results_list):
    """Detect person (optionally male/female) wearing any clothing of given color"""
    tracker = CooldownTracker()
    frame_id = 0
    print(f"  Searching: {gender if gender else 'person'} wearing {color_name} (any clothing)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        world_model.set_classes(["person"])
        res = world_model(frame, conf=0.35, imgsz=640)
        if res[0].boxes is None: continue

        for box in res[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            if (x2-x1)*(y2-y1) < AREA_THRESHOLD: continue

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0: continue

            # Gender check if needed
            if gender:
                # Detect gender from face (reuse your predict_gender_deepface)
                face_crop = person_crop[:int(person_crop.shape[0]*0.3), :]  # top 30% for face
                if face_crop.size > 0:
                    detected_gender = predict_gender_deepface(face_crop)
                    if detected_gender != gender:
                        continue

            # Torso region for color
            h, w = person_crop.shape[:2]
            torso = person_crop[int(h*0.25):int(h*0.65), int(w*0.15):int(w*0.85)]
            if torso.size == 0:
                torso = person_crop

            target_ratio = color_ratio(torso, color_name)
            if target_ratio < 0.12:   # 20% area of that color
                continue

            label = f"{gender + ' ' if gender else ''}person wearing {color_name} clothes"
            key = f"person_color_{x1//50}_{y1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            img_path = save_detection(frame, [x1, y1, x2, y2], label, frame_id, "person_color", (0, 200, 255))
            results_list.append({
                "object": label,
                "color": color_name,
                "gender": gender,
                "bbox": [x1, y1, x2, y2],
                "image_path": img_path,
                "timestamp": frame_id
            })








def detect_bags_in_frame(frame, bag_type, conf=0.25):
    """
    Detect bags using YOLOv8 COCO classes.
    Returns list of [x1, y1, x2, y2] boxes.
    """
    # car_model = your existing YOLO("yolov8n.pt") instance
    class_map = {
        'backpack': [24],
        'handbag':  [26],
        'bag':      [24, 26],
        'suitcase': [28],
    }
    class_ids = class_map.get(bag_type.lower(), [24, 26])
 
    results = car_model(frame, classes=class_ids, conf=conf)
    boxes = []
    if results[0].boxes is not None:
        for b in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, b)
            if (x2 - x1) * (y2 - y1) < 320:
                continue
            boxes.append([x1, y1, x2, y2])

    # Fallback with YOLOWorld for CCTV angles where COCO bag classes get missed.
    world_labels = {
        "bag": ["bag", "handbag", "backpack"],
        "handbag": ["handbag", "purse", "shoulder bag"],
        "backpack": ["backpack", "school bag", "rucksack"],
    }.get(bag_type.lower(), ["bag", "handbag", "backpack"])

    for wl in world_labels:
        world_model.set_classes([wl])
        wres = world_model(frame, conf=0.22, imgsz=640)
        if wres[0].boxes is None:
            continue
        for b in wres[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, b)
            area = (x2 - x1) * (y2 - y1)
            if area < 300:
                continue
            candidate = [x1, y1, x2, y2]
            if not any(iou(candidate, ub) > 0.55 for ub in boxes):
                boxes.append(candidate)

    return boxes

def clothing_belongs_to_person_loose(p_box, c_box):
    # bas center clothing ka person ke andar ho aur thoda overlap ho
    px1,py1,px2,py2 = p_box
    cx1,cy1,cx2,cy2 = c_box
    ccx = (cx1+cx2)/2
    ccy = (cy1+cy2)/2
    if not (px1 <= ccx <= px2 and py1 <= ccy <= py2):
        return False
    iou_val = iou(p_box, c_box)
    return iou_val >= 0.05   # 5% overlap enough


def parse_prompt(prompt: str):
    p = prompt.strip().lower()
    # Image-only "auto detect" support: if user doesn't give a prompt,
    # run a sensible default detector set instead of mis-parsing.
    if not p:
        return {"mode": "auto", "prompt": prompt}
    if p in {"auto", "auto detect", "autodetect", "detect"}:
        return {"mode": "auto", "prompt": prompt}
    p = re.sub(r'\bauto\s*rickshaw\b', 'auto rickshaw', p)
    p = re.sub(r'\btuk[\s-]?tuk\b', 'tuk tuk', p)
    # Electric-rickshaw typo normalization (prevents fallback to generic YOLOWorld mode)
    p = re.sub(r'\be[\s-]?ricks?h?a?w\b', 'e-rickshaw', p)
    p = re.sub(r'\be[\s-]?riks?h?a?w\b', 'e-rickshaw', p)
    p = re.sub(r'\be[\s-]?rick?saw\b', 'e-rickshaw', p)
    p = re.sub(r'\be[\s-]?ricks?w\b', 'e-rickshaw', p)
    p = re.sub(r'\be[\s-]?riks?w\b', 'e-rickshaw', p)
    p = re.sub(r'\belectric[\s-]?riks?h?a?w\b', 'electric rickshaw', p)
    p = re.sub(r'\belectric[\s-]?rick?saw\b', 'electric rickshaw', p)
    p = re.sub(r'\belectric[\s-]?riks?w\b', 'electric rickshaw', p)
    # Common typo normalization for clothing prompts
    p = re.sub(r'\bsaree+\b', 'saree', p)
    p = re.sub(r'\bsare\b', 'saree', p)
    p = re.sub(r'\bsarii+\b', 'sari', p)
    p = re.sub(r'\bsalwar\s*sit+\b', 'salwar suit', p)
    p = re.sub(r'\bsalwar\s*sut+\b', 'salwar suit', p)
    p = re.sub(r'\bsalwar\s*su[i1]t+\b', 'salwar suit', p)
    p = re.sub(r'\bsalwar\s*kameez\b', 'salwar suit', p)
    subject_words = r'person|man|men|woman|women|male|female'

    colors = r'red|blue|black|white|green|yellow|orange|purple|pink|silver|grey|gray|brown|olive|olive green'
    clothing_items = r'shirt|tshirt|t-shirt|jacket|coat|jeans|saree|sare|sari|salwar\s*suit|kurti|pant|top|dress|sweater|hoodie|blazer|suit|kurta|lehenga|dupatta'
   
    clothing_items = (
    r'shirt|t-shirt|tshirt|jacket|coat|jeans|saree|sare|sari|salwar\s*suit|'
    r'kurti|pant|top|dress|sweater|hoodie|blazer|suit|kurta|'
    r'lehenga|dupatta|trousers|uniform'
)



    colors = (
    r'red|blue|black|white|green|yellow|orange|purple|pink|'
    r'silver|grey|gray|brown|olive|navy|maroon|cream|beige|cyan|magenta'
)





    # ========================================
    # 1. CLOTHING WITH COLOR (exact)
    # ========================================
    m = re.search(rf'(?:{subject_words})\s+(?:wearing|in|with)\s+(?:a\s+|an\s+)?({colors})\s+(?:a\s+|an\s+)?({clothing_items})', p)
    if m:
        clothing = normalize_clothing_type(m.group(2))
        return {"mode": "person_clothing_color", "color": m.group(1), "clothing_type": clothing, "prompt": prompt}

    # ========================================
    # 2. CLOTHING WITHOUT COLOR
    # ========================================
    m = re.search(rf'(?:{subject_words})\s+(?:wearing|in|with)\s+(?:a\s+|an\s+)?({clothing_items})', p)
    if m:
        clothing = normalize_clothing_type(m.group(1))
        return {"mode": "person_clothing_any", "clothing_type": clothing, "prompt": prompt}

    # ========================================
    # 3. HELMET PATTERNS (color + any / only any)
    # ========================================
    m = re.search(rf'person\s+(?:with|have|has|wearing|carrying)\s+({colors})\s+helmet', p)
    if m:
        return {"mode": "person_helmet_color", "color": m.group(1), "prompt": prompt}
    m = re.search(rf'person\s+({colors})\s+helmet', p)
    if m:
        return {"mode": "person_helmet_color", "color": m.group(1), "prompt": prompt}
    m = re.search(r'person\s+(?:with|have|has|wearing|carrying)\s+helmet', p)
    if m:
        return {"mode": "person_helmet_any", "prompt": prompt}

    # ========================================
    # 4. BAG PATTERNS (color + bag, bag only)
    # ========================================
    m = re.search(rf'person\s+(?:with|have|has|wearing|carrying)\s+({colors})\s+(bag|backpack|handbag)', p)
    if m:
        return {"mode": "person_bag_color", "color": m.group(1), "bag_type": m.group(2), "prompt": prompt}
    m = re.search(rf'person\s+({colors})\s+(bag|backpack|handbag)', p)
    if m:
        return {"mode": "person_bag_color", "color": m.group(1), "bag_type": m.group(2), "prompt": prompt}
    m = re.search(rf'person\s+with\s+({colors})\s+bag', p)
    if m:
        return {"mode": "person_bag_color", "color": m.group(1), "bag_type": "bag", "prompt": prompt}
    m = re.search(rf'person\s+with\s+({colors})\s+backpack', p)
    if m:
        return {"mode": "person_bag_color", "color": m.group(1), "bag_type": "backpack", "prompt": prompt}
    m = re.search(rf'person\s+with\s+({colors})\s+handbag', p)
    if m:
        return {"mode": "person_bag_color", "color": m.group(1), "bag_type": "handbag", "prompt": prompt}
    m = re.search(r'person\s+(?:with|have|has|wearing|carrying)\s+(bag|backpack|handbag)', p)
    if m:
        return {"mode": "person_bag_any", "bag_type": m.group(1), "prompt": prompt}
    m = re.search(r'person\s+carrying\s+(bag|backpack|handbag)', p)
    if m:
        return {"mode": "person_bag_any", "bag_type": m.group(1), "prompt": prompt}
    m = re.search(r'person\s+have\s+(bag|backpack|handbag)', p)
    if m:
        return {"mode": "person_bag_any", "bag_type": m.group(1), "prompt": prompt}

    # ========================================
    # 5. SHOES PATTERNS — UPDATED
    # ========================================
    shoe_types = r'shoes|sneakers|boots|sandals|footwear|chappal|chappals'
   
    m = re.search(rf'person\s+(?:with|have|has|wearing|carrying)\s+({colors})\s+({shoe_types})', p)
    if m:
        return {"mode": "person_shoes_color", "color": m.group(1), "shoe_type": m.group(2), "prompt": prompt}
   
    m = re.search(rf'person\s+({colors})\s+({shoe_types})', p)
    if m:
        return {"mode": "person_shoes_color", "color": m.group(1), "shoe_type": m.group(2), "prompt": prompt}
   
    m = re.search(rf'person\s+(?:with|have|has|wearing|carrying)\s+({shoe_types})', p)
    if m:
        return {"mode": "person_shoes_any", "shoe_type": m.group(1), "prompt": prompt}
   
    m = re.search(rf'person\s+have\s+({shoe_types})', p)
    if m:
        return {"mode": "person_shoes_any", "shoe_type": m.group(1), "prompt": prompt}
   
    m = re.search(rf'person\s+wearing\s+({shoe_types})', p)
    if m:
        shoe = m.group(1)
        # "chappal" → sandals
        if shoe in ("chappal", "chappals"):
            shoe = "sandals"
        return {"mode": "person_shoes_any", "shoe_type": shoe, "prompt": prompt}

    # ========================================
    # 6. PERSON WITHOUT HELMET
    # ========================================
    m = re.search(r'person\s+(?:without|without\s+a|without\s+any|no|not\s+having)\s+helmet', p)
    if m:
        return {"mode": "person_without_helmet", "prompt": prompt}
    m = re.search(r'person\s+no\s+helmet', p)
    if m:
        return {"mode": "person_without_helmet", "prompt": prompt}

    # ========================================
    # 7. TRIPLE RIDING
    # ========================================
    # Explicit numeric patterns: "3+ person on bike", "3 + person on scooter", etc.
    if re.search(r'(?:\b3\s*\+?\s*(?:person|people)\s+on\b|\bmore\s+than\s+(?:2|two)\s+(?:person|people)\s+on\b)', p):
        vehicle_type = "two_wheeler"
        if re.search(r'\b(bike|bikes|motorcycle|motorcycles|motorbike|motorbikes)\b', p):
            vehicle_type = "bike"
        elif re.search(r'\b(scooter|scooters|activa|moped|mopeds|vespa)\b', p):
            vehicle_type = "scooter"
        return {"mode": "triple_riding", "vehicle_type": vehicle_type, "prompt": prompt}

    if re.search(r'triple\s*riding', p):
        vehicle_type = "two_wheeler"
        if "bike" in p or "motorcycle" in p or "motorbike" in p:
            vehicle_type = "bike"
        elif "scooter" in p or "activa" in p or "moped" in p:
            vehicle_type = "scooter"
        return {"mode": "triple_riding", "vehicle_type": vehicle_type, "prompt": prompt}
    # ... (baaki triple riding patterns tujhe apne code se copy karne hain – maine shorten kiya)
    # NOTE: Tu yahan apne existing triple riding patterns paste kar de

    # ========================================
    # 8. PERSON ON VEHICLE WITH COLOR
    # ========================================
    vehicles = '|'.join(sorted(VEHICLE_ALIASES.keys(), key=len, reverse=True))
    m = re.search(rf'person\s+on\s+(?:a\s+)?({colors})\s+({vehicles})', p)
    if m:
        return {"mode": "person_vehicle_color", "vehicle_word": m.group(2), "vehicle_info": VEHICLE_ALIASES[m.group(2)], "color": m.group(1), "prompt": prompt}

    # ========================================
    # 9. PERSON ON VEHICLE (NO COLOR)
    # ========================================
    m = re.search(rf'person\s+on\s+(?:a\s+)?({vehicles})', p)
    if m and m.group(1) in VEHICLE_ALIASES:
        return {"mode": "person_vehicle", "vehicle_word": m.group(1), "vehicle_info": VEHICLE_ALIASES[m.group(1)], "prompt": prompt}

    # ========================================
    # 10. TEXT SEARCH
    # ========================================
    m = re.search(r'(text|poster|word|search)\s+(.+)', p)
    if m:
        return {"mode": "text_search", "target_text": m.group(2).strip().upper(), "prompt": prompt}

    # ========================================
    # 11. GENDER + COLOR (generic)
    # ========================================
    m = re.search(rf'(male|female)\s+(?:wearing|in)\s+({colors})', p)
    if m:
        return {"mode": "gender_color", "gender": m.group(1), "color": m.group(2), "prompt": prompt}

    # ========================================
    # 12. COLOR + VEHICLE (color_object)
    # ========================================
    detected_color = extract_color_from_prompt(p)
    any_color = bool(re.search(r'\bany[\s_]?colou?r\b', p))
    veh_word, v_info = find_vehicle_in_prompt(p)
    if detected_color and veh_word:
        return {"mode": "color_object", "color": detected_color, "vehicle_word": veh_word, "vehicle_info": v_info, "prompt": prompt}
    if any_color and veh_word:
        return {"mode": "color_object", "color": None, "vehicle_word": veh_word, "vehicle_info": v_info, "prompt": prompt}

    # ========================================
    # 13. GENERIC COLOR-ONLY (FALLBACK) – YAHI SABSE PEECHE
    # ========================================
    m = re.search(rf'(?:person|male|female)\s+(?:wearing|in)\s+(?:a\s+|an\s+)?({colors})', p)
    if m:
        color = m.group(1)
        gender = None
        if 'male' in p:
            gender = 'male'
        elif 'female' in p:
            gender = 'female'
        return {"mode": "person_color_only", "color": color, "gender": gender, "prompt": prompt}

    # ========================================
    # 14. PLATE DETECTION
    # ========================================
    plate_info = detect_plate_prompt_type(prompt.strip())
    if plate_info["type"] != "not_plate":
        return {"mode": "plate", "plate_info": plate_info, "prompt": prompt}

    # ========================================
    # 15. FALLBACK
    # ========================================
    if "person" in p:
        found = [kw for kw in PERSON_ATTRIBUTE_KEYWORDS if kw in p]
        if found:
            return {"mode": "person_attribute", "attributes": found, "prompt": prompt}
    return {"mode": "yoloworld", "prompt": prompt}




def clothing_belongs_to_person_strict(p_box, c_box, clothing_type="shirt"):
    px1, py1, px2, py2 = p_box
    cx1, cy1, cx2, cy2 = c_box

    ph = py2 - py1
    pw = px2 - px1
    ch = cy2 - cy1
    cw = cx2 - cx1

    if ph <= 0 or pw <= 0 or ch <= 0 or cw <= 0:
        return False

    ccx = (cx1 + cx2) / 2
    ccy = (cy1 + cy2) / 2

    # 1. Clothing center X — person ke andar (5% margin)
    if not (px1 - pw * 0.05 <= ccx <= px2 + pw * 0.05):
        return False

    # 2. Clothing center Y — clothing type se decide hoga
    UPPER = {"shirt","tshirt","t-shirt","jacket","coat",
             "sweater","hoodie","kurta","kurti","blazer","suit","top"}
    LOWER = {"jeans","pant","trousers","lehenga"}

    rel_center_y = (ccy - py1) / ph

    if clothing_type.lower() in UPPER:
        if not (0.10 <= rel_center_y <= 0.72):
            return False
    elif clothing_type.lower() in LOWER:
        if not (0.42 <= rel_center_y <= 0.95):
            return False
    else:  # dress, saree
        if not (0.15 <= rel_center_y <= 0.85):
            return False

    # 3. Height check — relaxed
    if not (0.15 * ph <= ch <= 0.85 * ph):
        return False

    # 4. Width check — relaxed
    if not (0.25 * pw <= cw <= 1.05 * pw):
        return False

    # 5. Overlap — dual check
    overlap_x    = max(0, min(px2, cx2) - max(px1, cx1))
    overlap_y    = max(0, min(py2, cy2) - max(py1, cy1))
    overlap_area = overlap_x * overlap_y
    cloth_area   = cw * ch
    person_area  = pw * ph

    # Person area ka kam se kam 8% overlap hona chahiye
    if overlap_area < person_area * 0.08:
        return False

    # Clothing ka kam se kam 50% person ke andar hona chahiye
    if cloth_area > 0 and overlap_area / cloth_area < 0.50:
        return False

    return True

def run_person_clothing_any_mode(cap, clothing_type, results_list):
    """
    Detect: person wearing [clothing_type] — WITHOUT color filter
    Strategy: YOLOWorld se person detect karo, phir clothing detect karo,
    spatial check se match karo.
    """
    tracker  = CooldownTracker()
    frame_id = 0
    clothing_type = normalize_clothing_type(clothing_type)

    # Clothing ke liye best YOLOWorld labels
    yolo_labels = CLOTHING_YOLO_LABELS.get(
        clothing_type.lower(),
        [clothing_type]  # fallback: original word
    )
    print(f"  Searching: person wearing {clothing_type}")
    print(f"  YOLOWorld labels: {yolo_labels}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        # Step 1: Detect persons
        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.40, imgsz=640)
        if res_p[0].boxes is None:
            continue
        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()

        if len(p_boxes) == 0:
            continue

        # Step 2: Detect clothing using multiple labels
        clothing_boxes = []
        for label in yolo_labels:
            world_model.set_classes([label])
            res_c = world_model(frame, conf=0.12, imgsz=640)
            if res_c[0].boxes is not None:
                for box in res_c[0].boxes.xyxy.cpu().numpy():
                    clothing_boxes.append(list(map(int, box)))

        # Deduplicate clothing boxes
        unique_cloth = []
        for cb in clothing_boxes:
            if not any(iou(cb, ub) > 0.5 for ub in unique_cloth):
                unique_cloth.append(cb)

        # Step 3: Person-crop fallback with same clothing labels for recall.
        # Still keeps exact clothing type by querying only requested labels.
        if not unique_cloth:
            for pbox in p_boxes:
                px1, py1, px2, py2 = map(int, pbox)
                if (px2 - px1) * (py2 - py1) < AREA_THRESHOLD:
                    continue
                person_crop = frame[py1:py2, px1:px2]
                if person_crop.size == 0:
                    continue

                found_local = False
                for label in yolo_labels:
                    world_model.set_classes([label])
                    lc = world_model(person_crop, conf=0.10, imgsz=512)
                    if lc[0].boxes is None:
                        continue
                    for b in lc[0].boxes.xyxy.cpu().numpy():
                        cx1, cy1, cx2, cy2 = map(int, b)
                        # local box size sanity
                        if (cx2 - cx1) * (cy2 - cy1) < 220:
                            continue
                        # convert local -> frame coords
                        gx1, gy1, gx2, gy2 = px1 + cx1, py1 + cy1, px1 + cx2, py1 + cy2
                        if not person_clothing_spatial_check([px1, py1, px2, py2], [gx1, gy1, gx2, gy2], clothing_type):
                            continue
                        key = f"cloth_any_{px1 // 50}_{py1 // 50}"
                        if tracker.should_skip(key, frame_id):
                            continue
                        tracker.update(key, frame_id)
                        label_text = f"person wearing {clothing_type}"
                        ann = frame.copy()
                        draw_box(ann, px1, py1, px2, py2, "person", (0, 255, 0))
                        draw_box(ann, gx1, gy1, gx2, gy2, clothing_type, (255, 80, 0))
                        img_path = os.path.join(SAVE_DIR, f"cloth_any_{get_session_id()}_{frame_id}.jpg")
                        cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        results_list.append({
                            "object":        label_text,
                            "clothing_type": clothing_type,
                            "person_bbox":   [px1, py1, px2, py2],
                            "clothing_bbox": [gx1, gy1, gx2, gy2],
                            "image_path":    img_path,
                            "timestamp":     frame_id,
                            "method":        "person_crop_label_match"
                        })
                        found_local = True
                        break
                    if found_local:
                        break

                # No loose ethnic fallback in strict mode; avoids tshirt->saree false positives.
                if (not found_local) and clothing_type in {"saree", "salwar suit"} and not STRICT_PROMPT_MATCH:
                    person_h, person_w = person_crop.shape[:2]
                    full_body = person_crop[int(person_h * 0.08):int(person_h * 0.96), int(person_w * 0.08):int(person_w * 0.92)]
                    if full_body.size == 0:
                        continue
                    if person_h < 80 or person_w < 30:
                        continue
                    key = f"cloth_any_eth_{px1 // 50}_{py1 // 50}"
                    if tracker.should_skip(key, frame_id):
                        continue
                    tracker.update(key, frame_id)
                    label_text = f"person wearing {clothing_type}"
                    ann = frame.copy()
                    draw_box(ann, px1, py1, px2, py2, label_text, (0, 200, 100))
                    img_path = os.path.join(SAVE_DIR, f"cloth_any_eth_{get_session_id()}_{frame_id}.jpg")
                    cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    results_list.append({
                        "object":        label_text,
                        "clothing_type": clothing_type,
                        "person_bbox":   [px1, py1, px2, py2],
                        "image_path":    img_path,
                        "timestamp":     frame_id,
                        "method":        "ethnic_person_fallback"
                    })
            continue

        # Step 4: Match clothing boxes to person boxes
        for cbox in unique_cloth:
            cx1, cy1, cx2, cy2 = cbox
            cloth_area = (cx2 - cx1) * (cy2 - cy1)
            if cloth_area < 400:
                continue

            best_p_idx = -1
            best_score = 0.0

            for p_idx, pbox in enumerate(p_boxes):
                px1, py1, px2, py2 = map(int, pbox)
                if (px2 - px1) * (py2 - py1) < AREA_THRESHOLD:
                    continue
                if person_clothing_spatial_check(
                    [px1, py1, px2, py2], cbox, clothing_type
                ):
                    iou_val = iou([px1, py1, px2, py2], cbox)
                    if iou_val > best_score:
                        best_score = iou_val
                        best_p_idx = p_idx

            if best_p_idx == -1:
                continue

            pbox = p_boxes[best_p_idx]
            px1, py1, px2, py2 = map(int, pbox)

            key = f"cloth_any_{px1 // 50}_{py1 // 50}"
            if tracker.should_skip(key, frame_id):
                continue
            tracker.update(key, frame_id)

            label_text = f"person wearing {clothing_type}"
            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, "person", (0, 255, 0))
            draw_box(ann, cx1, cy1, cx2, cy2, clothing_type, (255, 80, 0))
            img_path = os.path.join(
                SAVE_DIR,
                f"cloth_any_{get_session_id()}_{frame_id}.jpg"
            )
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
            results_list.append({
                "object":        label_text,
                "clothing_type": clothing_type,
                "person_bbox":   [px1, py1, px2, py2],
                "clothing_bbox": cbox,
                "image_path":    img_path,
                "timestamp":     frame_id,
                "method":        "yolo_match"
            })






def clothing_belongs_to_person(p_box, c_box):
    px1,py1,px2,py2=p_box; cx1,cy1,cx2,cy2=c_box
    ccx=(cx1+cx2)/2; ccy=(cy1+cy2)/2
    pw=px2-px1; ph=py2-py1
    return (px1<=ccx<=px2 and py1+ph*0.2<=ccy<=py1+ph*0.7)


def run_person_clothing_color_mode(cap, clothing_type, color_name, results_list):
    tracker  = CooldownTracker()
    frame_id = 0
    clothing_type = normalize_clothing_type(clothing_type)
    color_name = normalize_color_name(color_name)

    yolo_labels = CLOTHING_YOLO_LABELS.get(clothing_type.lower(), [clothing_type])
    color_threshold = get_color_threshold(color_name)

    print(f"  Searching: person wearing {color_name} {clothing_type}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.40, imgsz=640)
        if res_p[0].boxes is None:
            continue
        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
        if len(p_boxes) == 0:
            continue

        clothing_boxes = []
        for label in yolo_labels:
            world_model.set_classes([label])
            res_c = world_model(frame, conf=0.12, imgsz=640)
            if res_c[0].boxes is not None:
                for box in res_c[0].boxes.xyxy.cpu().numpy():
                    clothing_boxes.append(list(map(int, box)))

        unique_cloth = []
        for cb in clothing_boxes:
            if not any(iou(cb, ub) > 0.5 for ub in unique_cloth):
                unique_cloth.append(cb)

        matched_this_frame = set()

        # ── METHOD A: YOLOWorld clothing boxes ──
        for cbox in unique_cloth:
            cx1, cy1, cx2, cy2 = cbox
            if (cx2 - cx1) * (cy2 - cy1) < 400:
                continue

            cloth_crop = frame[cy1:cy2, cx1:cx2]
            if cloth_crop.size == 0:
                continue

            color_ok, target_ratio = is_color_dominant(cloth_crop, color_name, color_threshold * 0.78)
            if not color_ok:
                continue

            best_p_idx = -1
            best_score = 0.0
            for p_idx, pbox in enumerate(p_boxes):
                px1, py1, px2, py2 = map(int, pbox)
                if (px2 - px1) * (py2 - py1) < AREA_THRESHOLD:
                    continue
                if person_clothing_spatial_check([px1, py1, px2, py2], cbox, clothing_type):
                    iou_val = iou([px1, py1, px2, py2], cbox)
                    if iou_val > best_score:
                        best_score = iou_val
                        best_p_idx = p_idx

            if best_p_idx == -1:
                continue

            pbox = p_boxes[best_p_idx]
            px1, py1, px2, py2 = map(int, pbox)

            key = f"cloth_col_{px1 // 50}_{py1 // 50}"
            if tracker.should_skip(key, frame_id):
                continue
            tracker.update(key, frame_id)
            matched_this_frame.add(best_p_idx)

            label_text = f"person wearing {color_name} {clothing_type}"
            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, "person", (0, 255, 0))
            draw_box(ann, cx1, cy1, cx2, cy2, f"{color_name} {clothing_type}", (0, 100, 255))
            img_path = os.path.join(SAVE_DIR, f"cloth_color_{get_session_id()}_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
            results_list.append({
                "object":        label_text,
                "clothing_type": clothing_type,
                "color":         color_name,
                "color_ratio":   round(target_ratio, 3),
                "person_bbox":   [px1, py1, px2, py2],
                "clothing_bbox": cbox,
                "image_path":    img_path,
                "timestamp":     frame_id,
                "method":        "yolo_color_match"
            })

        # ── METHOD B: Person-crop label + color fallback ──
        for p_idx, pbox in enumerate(p_boxes):
            if p_idx in matched_this_frame:
                continue

            px1, py1, px2, py2 = map(int, pbox)
            if (px2 - px1) * (py2 - py1) < AREA_THRESHOLD:
                continue
            person_crop = frame[py1:py2, px1:px2]
            if person_crop.size == 0 or is_blurry(person_crop):
                continue

            found_local = False
            target_ratio = 0.0
            fb_box = None
            for label in yolo_labels:
                world_model.set_classes([label])
                lc = world_model(person_crop, conf=0.10, imgsz=512)
                if lc[0].boxes is None:
                    continue
                for b in lc[0].boxes.xyxy.cpu().numpy():
                    cx1, cy1, cx2, cy2 = map(int, b)
                    if (cx2 - cx1) * (cy2 - cy1) < 220:
                        continue
                    gx1, gy1, gx2, gy2 = px1 + cx1, py1 + cy1, px1 + cx2, py1 + cy2
                    if not person_clothing_spatial_check([px1, py1, px2, py2], [gx1, gy1, gx2, gy2], clothing_type):
                        continue
                    local_crop = frame[gy1:gy2, gx1:gx2]
                    if local_crop.size == 0:
                        continue
                    color_ok, ratio = is_color_dominant(local_crop, color_name, threshold=color_threshold * 0.72)
                    if not color_ok:
                        continue
                    target_ratio = ratio
                    fb_box = [gx1, gy1, gx2, gy2]
                    found_local = True
                    break
                if found_local:
                    break
            if not found_local:
                # Ethnic fallback for color mode (saree/salwar suit):
                # when explicit clothing box misses, use full-body color validation.
                if clothing_type not in {"saree", "salwar suit"}:
                    continue
                person_h, person_w = person_crop.shape[:2]
                color_region = person_crop[int(person_h * 0.12):int(person_h * 0.95), int(person_w * 0.10):int(person_w * 0.90)]
                if color_region.size == 0:
                    continue
                color_ok, ratio = is_color_dominant(color_region, color_name, threshold=color_threshold * 0.68)
                if not color_ok:
                    continue
                if not ethnic_color_region_guard(person_crop, color_name, clothing_type):
                    continue
                target_ratio = ratio
                fb_box = [px1, py1, px2, py2]

            key = f"cloth_col_fb_{px1 // 50}_{py1 // 50}"
            if tracker.should_skip(key, frame_id):
                continue
            tracker.update(key, frame_id)

            label_text = f"person wearing {color_name} {clothing_type}"
            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, label_text, (0, 180, 255))
            if fb_box:
                draw_box(ann, fb_box[0], fb_box[1], fb_box[2], fb_box[3], f"{color_name} {clothing_type}", (255, 80, 0))
            img_path = os.path.join(SAVE_DIR, f"cloth_color_fb_{get_session_id()}_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
            results_list.append({
                "object":        label_text,
                "clothing_type": clothing_type,
                "color":         color_name,
                "color_ratio":   round(target_ratio, 3),
                "person_bbox":   [px1, py1, px2, py2],
                "clothing_bbox": fb_box if fb_box else [],
                "image_path":    img_path,
                "timestamp":     frame_id,
                "method":        "person_crop_label_color_fallback"
            })








def run_text_search_mode(cap, target_text, results_list):
    tracker  = CooldownTracker()
    frame_id = 0

    print(f"Searching text: {target_text}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        # OCR on full frame
        results = reader.readtext(frame, detail=1)

        for (bbox, text, conf) in results:
            clean = re.sub(r'[^A-Z0-9]', '', text.upper())

            if len(clean) < 3:
                continue

            # match text
            if target_text in clean:
                pts = np.array(bbox).astype(int)
                x1, y1 = pts[0]
                x2, y2 = pts[2]

                key = f"text_{x1//50}_{y1//50}"
                if tracker.should_skip(key, frame_id):
                    continue
                tracker.update(key, frame_id)

                label = f"text: {clean}"

                ann = frame.copy()
                draw_box(ann, x1, y1, x2, y2, label, (0,255,255))

                img_path = os.path.join(SAVE_DIR, f"text_{get_session_id()}_{frame_id}.jpg")
                cv2.imwrite(img_path, ann)

                results_list.append({
                    "object": "text",
                    "detected_text": clean,
                    "target": target_text,
                    "bbox": [x1, y1, x2, y2],
                    "image_path": img_path,
                    "timestamp": frame_id
                })




def bag_belongs_to_person_strict(p_box, b_box):
    px1, py1, px2, py2 = p_box
    bx1, by1, bx2, by2 = b_box

    # Bag ka center
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2

    # 1. Bag ka center person ke box ke andar hona chahiye (covers side bags)
    if not (px1 <= bcx <= px2 and py1 <= bcy <= py2):
        # But allow slightly outside (e.g., backpack sticking out)
        if not (px1 - (px2-px1)*0.2 <= bcx <= px2 + (px2-px1)*0.2):
            return False
        if not (py1 <= bcy <= py2 + (py2-py1)*0.3):
            return False

    # 2. Bag person ke bahut upar nahi hona chahiye (head area)
    if bcy < py1 + (py2-py1)*0.2:  # 20% upper area is head
        return False

    # 3. Bag person ke bahut neeche bhi nahi (feet area)
    if bcy > py2 - (py2-py1)*0.1:
        return False

    # 4. Overlap area kam se kam 5% of person area
    person_area = (px2-px1)*(py2-py1)
    overlap = max(0, min(px2, bx2) - max(px1, bx1)) * max(0, min(py2, by2) - max(py1, by1))
    if overlap < person_area * 0.08:
        return False

    return True



def run_person_bag_any_mode(cap, bag_type, results_list):
    tracker  = CooldownTracker()
    frame_id = 0
    print(f"  Searching: person with ANY {bag_type}")
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue
 
        # Detect persons
        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.35, imgsz=640)
        if res_p[0].boxes is None:
            continue
 
        # Detect bags — any type
        bag_boxes = detect_bags_in_frame(frame, bag_type, conf=0.22)
        if not bag_boxes:
            continue
 
        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
 
        for bbox in bag_boxes:
            bx1, by1, bx2, by2 = bbox
            bag_cx = (bx1 + bx2) / 2
            bag_cy = (by1 + by2) / 2
            bag_w = bx2 - bx1
            bag_h = by2 - by1
 
            # Get actual bag color for label
            bag_crop = frame[by1:by2, bx1:bx2]
            actual_color = get_dominant_color_name(bag_crop) if bag_crop.size > 0 else "unknown"
 
            # Find person
            best_p_idx = -1
            best_overlap = 0.0
 
            for p_idx, pbox in enumerate(p_boxes):
                px1, py1, px2, py2 = map(int, pbox)
 
                # ✅ PERSON BOX SIZE GUARD
                if not is_valid_person_box([px1, py1, px2, py2], frame.shape):
                    continue
 
                pw = px2 - px1
                ph = py2 - py1
 
                # Bag center inside/near person
                in_person = (
                    px1 - pw * 0.20 <= bag_cx <= px2 + pw * 0.20 and
                    py1 <= bag_cy <= py2 + ph * 0.15
                )
                if not in_person:
                    continue
 
                # Bag not bigger than person
                if bag_w > pw * 0.80 or bag_h > ph * 0.70:
                    continue
 
                overlap_x = max(0, min(px2, bx2) - max(px1, bx1))
                overlap_y = max(0, min(py2, by2) - max(py1, by1))
                overlap = overlap_x * overlap_y
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_p_idx = p_idx
 
            if best_p_idx == -1:
                continue
 
            pbox = p_boxes[best_p_idx]
            px1, py1, px2, py2 = map(int, pbox)
 
            key = f"bag_any_{px1 // 50}_{py1 // 50}"
            if tracker.should_skip(key, frame_id):
                continue
            tracker.update(key, frame_id)
 
            label = f"person with {actual_color} {bag_type}"
            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, "person", (0, 255, 0))
            draw_box(ann, bx1, by1, bx2, by2, f"{actual_color} {bag_type}", (0, 200, 255))
 
            img_path = os.path.join(
                SAVE_DIR,
                f"bag_any_{get_session_id()}_{frame_id}.jpg"
            )
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
 
            results_list.append({
                "object":       label,
                "color":        actual_color,
                "bag_type":     bag_type,
                "person_bbox":  [px1, py1, px2, py2],
                "bag_bbox":     [bx1, by1, bx2, by2],
                "image_path":   img_path,
                "timestamp":    frame_id
            })

def debug_bag_color(image_path, bag_box, requested_color):
    """
    Print color ratios for a bag crop.
    Usage: debug_bag_color("detected_frames/bag_color_xxx.jpg",
                            [x1,y1,x2,y2], "yellow")
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read {image_path}")
        return
 
    bx1, by1, bx2, by2 = bag_box
    crop = img[by1:by2, bx1:bx2]
    if crop.size == 0:
        print("Empty crop")
        return
 
    print(f"\n=== Color Debug for {image_path} ===")
    print(f"Requested: {requested_color}")
    print(f"Bag box: {bag_box} ({bx2-bx1}x{by2-by1} px)")
    print()
 
    ratios = {}
    for cname in COLOR_RANGES:
        r = color_ratio(crop, cname)
        if r > 0.01:
            ratios[cname] = r
 
    ratios = dict(sorted(ratios.items(), key=lambda x: -x[1]))
    for cname, r in ratios.items():
        marker = " ← DOMINANT" if list(ratios.keys())[0] == cname else ""
        target = " ← REQUESTED" if cname == requested_color else ""
        print(f"  {cname:12s}: {r:.3f} ({r*100:.1f}%){marker}{target}")
 
    matched, ratio, actual = is_color_match_for_bag(crop, requested_color)
    print(f"\nResult: matched={matched}, ratio={ratio:.3f}, actual={actual}")
    print("=" * 40)



def detect_footwear_in_frame(frame, shoe_type="shoes", conf=0.30):
    classes = FOOTWEAR_CLASSES.get(shoe_type.lower(), ["closed toe shoes", "sneakers", "boots"])
   
    # Keywords jo sandals indicate karte hain — reject karo jab shoe_type != sandals
    SANDAL_KEYWORDS = {"sandal", "flip", "flop", "slipper", "open toe", "thong"}
   
    results = []
    for cls_name in classes:
        world_model.set_classes([cls_name])
        res = world_model(frame, conf=conf, imgsz=640)
        if res[0].boxes is None:
            continue
           
        for box_idx, box in enumerate(res[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            if area < 200:
                continue
           
            # ── SANDAL FILTER ──────────────────────────────────────
            # Agar shoe_type "sandals" nahi hai, to sandal-like
            # detections reject karo using visual check
            if shoe_type.lower() not in ("sandals", "footwear"):
                shoe_crop = frame[y1:y2, x1:x2]
                if shoe_crop.size > 0 and is_sandal_by_shape(shoe_crop):
                    print(f"  ⛔ Sandal shape detected — skipping (requested: {shoe_type})")
                    continue
            # ──────────────────────────────────────────────────────
           
            results.append(([x1, y1, x2, y2], cls_name))
   
    # Deduplicate
    final = []
    for b, lbl in results:
        if not any(iou(b, f[0]) > 0.5 for f in final):
            final.append((b, lbl))
   
    return final

def is_sandal_by_shape(shoe_crop):
    if shoe_crop is None or shoe_crop.size == 0:
        return False
   
    h, w = shoe_crop.shape[:2]
    if h < 10 or w < 10:
        return False
   
    hsv = cv2.cvtColor(shoe_crop, cv2.COLOR_BGR2HSV)
   
    # ── 🔥 ORANGE SHOE GUARD — check karo pehle ─────────────────
    # Agar crop mein significant orange hai → definitely NOT sandal
    orange_lower = np.array([5, 120, 100], dtype=np.uint8)
    orange_upper = np.array([25, 255, 255], dtype=np.uint8)
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    total_pixels = h * w
    orange_ratio = cv2.countNonZero(orange_mask) / (total_pixels + 1e-5)
   
    if orange_ratio > 0.20:  # 20%+ orange = shoe/sneaker, NOT sandal
        return False
   
    # ── RED SHOE GUARD ───────────────────────────────────────────
    red_mask1 = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160, 120, 100]), np.array([180, 255, 255]))
    red_ratio = cv2.countNonZero(cv2.bitwise_or(red_mask1, red_mask2)) / (total_pixels + 1e-5)
    if red_ratio > 0.20:
        return False
   
    # ── BLUE SHOE GUARD ──────────────────────────────────────────
    blue_mask = cv2.inRange(hsv, np.array([90, 80, 60]), np.array([140, 255, 255]))
    blue_ratio = cv2.countNonZero(blue_mask) / (total_pixels + 1e-5)
    if blue_ratio > 0.20:
        return False

    # ── GREEN SHOE GUARD ─────────────────────────────────────────
    green_mask = cv2.inRange(hsv, np.array([35, 80, 60]), np.array([85, 255, 255]))
    green_ratio = cv2.countNonZero(green_mask) / (total_pixels + 1e-5)
    if green_ratio > 0.20:
        return False
   
    # ── 1. Skin color detection ──────────────────────────────────
    # NARROWER skin range to avoid orange/red shoe false positives
    skin_lower = np.array([0, 20, 100], dtype=np.uint8)   # raised V min
    skin_upper = np.array([18, 150, 255], dtype=np.uint8)  # narrowed S max
    skin_mask1 = cv2.inRange(hsv, skin_lower, skin_upper)
   
    skin_lower2 = np.array([165, 20, 100], dtype=np.uint8)
    skin_upper2 = np.array([180, 150, 255], dtype=np.uint8)
    skin_mask2 = cv2.inRange(hsv, skin_lower2, skin_upper2)
   
    skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
    skin_pixels = cv2.countNonZero(skin_mask)
    skin_ratio = skin_pixels / (total_pixels + 1e-5)
   
    if skin_ratio > 0.30:  # raised threshold (was 0.25)
        return True
   
    # ── 2. Coverage check ────────────────────────────────────────
    gray = cv2.cvtColor(shoe_crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    solid_pixels = cv2.countNonZero(binary)
    coverage_ratio = solid_pixels / (total_pixels + 1e-5)
   
    if coverage_ratio < 0.30:  # lowered slightly (was 0.35)
        return True
   
    # ── 3. Aspect ratio + skin combo ─────────────────────────────
    aspect = w / (h + 1e-5)
    if aspect > 2.5 and skin_ratio > 0.20:
        return True
   
    return False


def shoes_belongs_to_person_strict(p_box, s_box):
    px1, py1, px2, py2 = p_box
    sx1, sy1, sx2, sy2 = s_box

    ph = py2 - py1
    pw = px2 - px1
    sh = sy2 - sy1
    sw = sx2 - sx1

    # 1. Shoes ka bottom person ke bottom ke paas (max 20% upar)
    if sy2 < py2 - ph * 0.2:
        return False

    # 2. Shoes ka top person ke bottom se zyada upar nahi hona chahiye
    if sy1 > py2:
        return False

    # 3. Shoes ki height person height ka 10% se 35% ke beech
    if not (0.10 * ph <= sh <= 0.45 * ph):
        return False

    # 4. Shoes ki width person width ka 15% se 50% ke beech
    if not (0.15 * pw <= sw <= 0.50 * pw):
        return False

    # 5. Overlap area kam se kam 5% of person area
    overlap_x = max(0, min(px2, sx2) - max(px1, sx1))
    overlap_y = max(0, min(py2, sy2) - max(py1, sy1))
    overlap_area = overlap_x * overlap_y
    person_area = pw * ph
    if overlap_area < person_area * 0.05:
        return False

    # 6. Center X person ke andar ya thoda bahar (paon alag)
    scx = (sx1 + sx2) / 2
    if not (px1 - pw*0.2 <= scx <= px2 + pw*0.2):
        return False

    return True


def is_plausible_shoe_box_for_person(p_box, s_box):
    """
    Extra anti-false-positive gate:
    - Shoe must lie in lower body region
    - Shoe size must be reasonable vs person size
    - Shoe shape shouldn't be extreme
    """
    px1, py1, px2, py2 = p_box
    sx1, sy1, sx2, sy2 = s_box

    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)
    sw = max(1, sx2 - sx1)
    sh = max(1, sy2 - sy1)
    s_area = sw * sh
    p_area = pw * ph

    # Shoe center should be in lower 40% of person.
    scy = (sy1 + sy2) / 2.0
    rel_y = (scy - py1) / ph
    if not (0.56 <= rel_y <= 1.05):
        return False

    # Area ratio bounds to reject large black objects (bags, coats, shadows).
    ratio = s_area / (p_area + 1e-6)
    if not (0.002 <= ratio <= 0.14):
        return False

    # Shape gate.
    aspect = sw / (sh + 1e-6)
    if not (0.35 <= aspect <= 4.6):
        return False

    return True












def run_person_shoes_any_mode(cap,shoe_type, results_list):
    tracker  = CooldownTracker()
    frame_id = 0
    shoe_type = normalize_shoe_type(shoe_type)
    print(f"  Searching: person with {shoe_type} (excluding sandals)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.40, imgsz=640)
        if res_p[0].boxes is None: continue

        # Use specific shoe classes — NOT sandals
        footwear_detections = detect_footwear_in_frame(frame, shoe_type, conf=0.30)
        if not footwear_detections: continue

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()

        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            if (px2 - px1) * (py2 - py1) < AREA_THRESHOLD: continue

            for sbox, shoe_label in footwear_detections:
                detected_type = canonicalize_detected_shoe_label(shoe_label)
                if shoe_type != "footwear" and detected_type != shoe_type:
                    continue
                sx1, sy1, sx2, sy2 = sbox
                if not shoes_belongs_to_person_strict([px1, py1, px2, py2], [sx1, sy1, sx2, sy2]):
                    continue
                if not is_plausible_shoe_box_for_person([px1, py1, px2, py2], [sx1, sy1, sx2, sy2]):
                    continue

                label = f"person with {shoe_type}"
                key = f"shoes_any_{px1//50}_{py1//50}"
                if tracker.should_skip(key, frame_id): continue
                tracker.update(key, frame_id)

                ann = frame.copy()
                draw_box(ann, px1, py1, px2, py2, "person", (0, 255, 0))
                draw_box(ann, sx1, sy1, sx2, sy2, shoe_label, (255, 0, 0))
                img_path = os.path.join(SAVE_DIR, f"shoes_any_{get_session_id()}_{frame_id}.jpg")
                cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])

                results_list.append({
                    "object": label,
                    "shoe_type": shoe_type,
                    "person_bbox": [px1, py1, px2, py2],
                    "shoes_bbox": [sx1, sy1, sx2, sy2],
                    "image_path": img_path,
                    "timestamp": frame_id
                })
                break  # one shoe match per person is enough


        # Fallback: if no shoe detected but color prompt might come? This mode is without color, so no fallback.

def run_person_shoes_color_mode(cap, color_name, shoe_type, results_list):
    tracker  = CooldownTracker()
    frame_id = 0
    color_name = normalize_color_name(color_name)
    shoe_type = normalize_shoe_type(shoe_type)
    print(f"  Searching: person with {color_name} {shoe_type}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.40, imgsz=640)
        if res_p[0].boxes is None: continue

        footwear_detections = detect_footwear_in_frame(frame, shoe_type, conf=0.18)  # lower conf for small/far shoes

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
        matched_person_idx = set()

        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            if (px2 - px1) * (py2 - py1) < AREA_THRESHOLD: continue

            for sbox, shoe_label in footwear_detections:
                detected_type = canonicalize_detected_shoe_label(shoe_label)
                if shoe_type != "footwear" and detected_type != shoe_type:
                    continue
                sx1, sy1, sx2, sy2 = sbox
                shoe_area = (sx2 - sx1) * (sy2 - sy1)
                if shoe_area < 100:
                    continue

                if not shoes_belongs_to_person_strict([px1, py1, px2, py2], [sx1, sy1, sx2, sy2]):
                    continue
                if not is_plausible_shoe_box_for_person([px1, py1, px2, py2], [sx1, sy1, sx2, sy2]):
                    continue

                shoes_crop = frame[sy1:sy2, sx1:sx2]
                if shoes_crop.size == 0:
                    continue

                # ── COLOR CHECK ──────────────────────────────────────────
                target_ratio = color_ratio(shoes_crop, color_name)

                # Shoes are small objects; keep per-color practical thresholds.
                bright_colors = {"orange", "red", "yellow", "pink", "green", "blue", "purple"}
                dark_colors = {"black", "brown", "navy", "grey", "gray"}
                if color_name in dark_colors:
                    threshold = 0.10
                elif color_name in {"red", "orange"}:
                    threshold = 0.05
                elif color_name in bright_colors:
                    threshold = 0.12
                else:
                    threshold = 0.14

                if target_ratio < threshold:
                    continue

                # Dominant color confirmation (balanced): target should be close to top color.
                best_color, best_ratio = color_name, target_ratio
                for c in COLOR_RANGES:
                    if c == color_name: continue
                    r = color_ratio(shoes_crop, c)
                    if r > best_ratio:
                        best_color = c
                        best_ratio = r

                # Strong-confusion protection
                if color_name == "red":
                    orange_ratio = color_ratio(shoes_crop, "orange")
                    if orange_ratio > target_ratio * 1.80:
                        continue
                if color_name == "orange":
                    red_ratio = color_ratio(shoes_crop, "red")
                    if red_ratio > target_ratio * 1.80:
                        continue

                # Reject only when competitor is clearly much stronger.
                if best_color != color_name:
                    if color_name in {"black", "white", "grey", "gray", "silver"}:
                        if best_ratio > target_ratio * 1.25:
                            continue
                    else:
                        if best_ratio > target_ratio * 1.55:
                            continue

                # ✅ MATCH
                label = f"person with {color_name} {shoe_type}"
                key = f"shoes_color_{px1//50}_{py1//50}"
                if tracker.should_skip(key, frame_id): continue
                tracker.update(key, frame_id)

                ann = frame.copy()
                draw_box(ann, px1, py1, px2, py2, "person", (0, 255, 0))
                draw_box(ann, sx1, sy1, sx2, sy2, f"{color_name} {shoe_label}", (255, 0, 0))
                img_path = os.path.join(SAVE_DIR, f"shoes_color_{get_session_id()}_{frame_id}.jpg")
                cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])

                results_list.append({
                    "object": label,
                    "color": color_name,
                    "shoe_type": shoe_type,
                    "person_bbox": [px1, py1, px2, py2],
                    "shoes_bbox": [sx1, sy1, sx2, sy2],
                    "image_path": img_path,
                    "timestamp": frame_id
                })
                matched_person_idx.add((px1 // 20, py1 // 20))
                break  # one shoe match per person

        # Fallback: if shoe detector misses, use lower-body foot region color.
        # Enable for all shoe prompts to reduce false negatives in CCTV videos.
        neutral_colors = {"black", "white", "grey", "gray", "silver"}
        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            pkey = (px1 // 20, py1 // 20)
            if pkey in matched_person_idx:
                continue
            if (px2 - px1) * (py2 - py1) < AREA_THRESHOLD:
                continue

            person_crop = frame[py1:py2, px1:px2]
            if person_crop.size == 0:
                continue

            ph, pw = person_crop.shape[:2]
            # Bottom region where shoes are expected.
            foot_region = person_crop[int(ph * 0.72):int(ph * 0.99), int(pw * 0.08):int(pw * 0.92)]
            if foot_region.size == 0:
                continue

            target_ratio = color_ratio(foot_region, color_name)
            max_other = 0.0
            for c in COLOR_RANGES:
                if c == color_name:
                    continue
                r = color_ratio(foot_region, c)
                if r > max_other:
                    max_other = r

            # Relaxed but controlled threshold for fallback region.
            thr = 0.08
            dom_mult = 1.55
            if color_name in neutral_colors:
                thr = 0.07
                dom_mult = 1.35
            elif color_name in {"red", "orange"}:
                thr = 0.045
                dom_mult = 1.90
            if target_ratio < thr:
                continue
            if max_other > target_ratio * dom_mult:
                continue

            # Neutral-color safety: require evidence in BOTH left and right foot areas.
            if color_name in neutral_colors:
                left = person_crop[int(ph * 0.78):int(ph * 0.98), int(pw * 0.08):int(pw * 0.45)]
                right = person_crop[int(ph * 0.78):int(ph * 0.98), int(pw * 0.55):int(pw * 0.92)]
                if left.size == 0 or right.size == 0:
                    continue
                l_ratio = color_ratio(left, color_name)
                r_ratio = color_ratio(right, color_name)
                if l_ratio < 0.05 or r_ratio < 0.05:
                    continue

            key = f"shoes_color_fb_{px1//50}_{py1//50}"
            if tracker.should_skip(key, frame_id):
                continue
            tracker.update(key, frame_id)

            label = f"person with {color_name} {shoe_type}"
            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, "person", (0, 255, 0))
            # draw approximated shoes region in full-frame coords
            sx1 = px1 + int((px2 - px1) * 0.10)
            sx2 = px1 + int((px2 - px1) * 0.90)
            sy1 = py1 + int((py2 - py1) * 0.78)
            sy2 = py1 + int((py2 - py1) * 0.98)
            draw_box(ann, sx1, sy1, sx2, sy2, f"{color_name} {shoe_type} (fb)", (255, 0, 0))
            img_path = os.path.join(SAVE_DIR, f"shoes_color_fb_{get_session_id()}_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])

            results_list.append({
                "object": label,
                "color": color_name,
                "shoe_type": shoe_type,
                "person_bbox": [px1, py1, px2, py2],
                "shoes_bbox": [sx1, sy1, sx2, sy2],
                "image_path": img_path,
                "timestamp": frame_id,
                "method": "foot_region_fallback"
            })


def shoes_belongs_to_person(p_box, s_box):
    px1,py1,px2,py2 = p_box
    sx1,sy1,sx2,sy2 = s_box
    scx = (sx1 + sx2) / 2
    scy = (sy1 + sy2) / 2
    pw = px2 - px1
    # Person's bottom region (lower 30%) should contain shoes
    return (px1 <= scx <= px2 and
            sy2 > py2 - (py2-py1)*0.7 and
            (sx2 - sx1) < pw * 1.2)

def run_person_bag_color_mode(cap, color_name, bag_type, results_list):
    tracker  = CooldownTracker()
    frame_id = 0
    print(f"  Searching: person with {color_name} {bag_type}")
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue
 
        frame_h, frame_w = frame.shape[:2]
 
        # Detect persons
        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.35, imgsz=640)
        if res_p[0].boxes is None:
            continue
 
        # Detect bags
        bag_boxes = detect_bags_in_frame(frame, bag_type, conf=0.25)
        if not bag_boxes:
            continue
 
        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
 
        for bbox in bag_boxes:
            bx1, by1, bx2, by2 = bbox
            bag_w = bx2 - bx1
            bag_h = by2 - by1
 
            # ── Bag crop color check ──────────────────────────
            bag_crop = frame[by1:by2, bx1:bx2]
            if bag_crop.size == 0:
                continue
 
            color_matched, bag_ratio, actual_color = is_color_match_for_bag(
                bag_crop, color_name
            )
 
            print(f"  Frame {frame_id}: bag at {bbox} | "
                  f"target={color_name}({bag_ratio:.2f}) | "
                  f"actual={actual_color} | match={color_matched}")
 
            if not color_matched:
                continue
 
            # ── Find person this bag belongs to ──────────────
            bag_cx = (bx1 + bx2) / 2
            bag_cy = (by1 + by2) / 2
 
            best_p_idx = -1
            best_overlap = 0.0
 
            for p_idx, pbox in enumerate(p_boxes):
                px1, py1, px2, py2 = map(int, pbox)
 
                # ✅ PERSON BOX SIZE GUARD (fixes Image 2 huge box)
                if not is_valid_person_box(
                    [px1, py1, px2, py2], frame.shape
                ):
                    print(f"  Skipping oversized person box: "
                          f"{px2-px1}x{py2-py1} vs frame {frame_w}x{frame_h}")
                    continue
 
                # Bag center inside person box (with 20% margin)
                pw = px2 - px1
                ph = py2 - py1
                in_person = (
                    px1 - pw * 0.20 <= bag_cx <= px2 + pw * 0.20 and
                    py1 <= bag_cy <= py2 + ph * 0.15
                )
                if not in_person:
                    continue
 
                # Bag size sanity: not bigger than person
                if bag_w > pw * 0.80 or bag_h > ph * 0.70:
                    continue
 
                # Compute overlap for best-match selection
                overlap_x = max(0, min(px2, bx2) - max(px1, bx1))
                overlap_y = max(0, min(py2, by2) - max(py1, by1))
                overlap = overlap_x * overlap_y
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_p_idx = p_idx
 
            if best_p_idx == -1:
                continue
 
            pbox = p_boxes[best_p_idx]
            px1, py1, px2, py2 = map(int, pbox)
 
            # Dedup
            key = f"bag_color_{px1 // 50}_{py1 // 50}"
            if tracker.should_skip(key, frame_id):
                continue
            tracker.update(key, frame_id)
 
            label = f"person with {color_name} {bag_type}"
            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, "person", (0, 255, 0))
            draw_box(ann, bx1, by1, bx2, by2,
                     f"{color_name} {bag_type} ({bag_ratio:.0%})", (255, 165, 0))
 
            img_path = os.path.join(
                SAVE_DIR,
                f"bag_color_{get_session_id()}_{frame_id}.jpg"
            )
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
 
            results_list.append({
                "object":       label,
                "color":        color_name,
                "actual_color": actual_color,
                "color_ratio":  round(bag_ratio, 3),
                "bag_type":     bag_type,
                "person_bbox":  [px1, py1, px2, py2],
                "bag_bbox":     [bx1, by1, bx2, by2],
                "image_path":   img_path,
                "timestamp":    frame_id
            })


def helmet_belongs_to_person_strict(p_box, h_box):
    """
    Check if helmet belongs to person using strict spatial criteria.
   
    Args:
        p_box: [x1, y1, x2, y2] of person
        h_box: [x1, y1, x2, y2] of helmet
   
    Returns:
        True if helmet is correctly placed on person's head
    """
    px1, py1, px2, py2 = p_box
    hx1, hy1, hx2, hy2 = h_box
   
    ph = py2 - py1          # person height
    pw = px2 - px1          # person width
   
    # Helmet center
    hcx = (hx1 + hx2) / 2
    hcy = (hy1 + hy2) / 2
   
    # 1. Helmet center must be within person's head region (top 40% of height)
    #    and horizontally within person's body (allow slight side offset)
    if not (px1 - pw*0.15 <= hcx <= px2 + pw*0.15):
        return False
    if not (py1 <= hcy <= py1 + ph * 0.40):
        return False
   
    # 2. Helmet bottom should be above shoulder level (not below 55% of height)
    if hy2 > py1 + ph * 0.55:
        return False
   
    # 3. Helmet should not be too far outside person's sides
    if hx1 < px1 - pw * 0.25 or hx2 > px2 + pw * 0.25:
        return False
   
    # 4. IoU (Intersection over Union) should be at least 5% (reasonable for small helmets)
    iou_val = iou(p_box, h_box)
    if iou_val < 0.05:
        return False
   
    # 5. Helmet width should be between 25% and 80% of person's width
    hw = hx2 - hx1
    if not (0.25 * pw <= hw <= 0.80 * pw):
        return False
   
    # 6. Helmet height should be reasonable (not too tall or too flat)
    hh = hy2 - hy1
    if hh < 5 or hh > ph * 0.50:
        return False
   
    return True

# 🔧 PERSON WITHOUT HELMET - COMPLETE FIX
# Replace your existing run_person_without_helmet_mode function with this

def run_person_without_helmet_mode(cap, results_list):
    """🪖 Detect: Person WITHOUT helmet on TWO-WHEELER only (strict - no pedestrians)"""
    tracker = CooldownTracker()
    frame_id = 0
    print(f"\n  🔍 Searching: person without helmet (two-wheeler only)\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        # ═════════════════════════════════════════════════════════════
        # STEP 1: PERSON DETECTION (RELAXED)
        # ═════════════════════════════════════════════════════════════
        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.35, imgsz=640)  # ✅ LOWERED from 0.45
        if res_p[0].boxes is None:
            continue

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
        if len(p_boxes) == 0:
            continue

        # ═════════════════════════════════════════════════════════════
        # STEP 2: HELMET DETECTION (BALANCED)
        # ═════════════════════════════════════════════════════════════
        world_model.set_classes(["helmet"])
        res_h = world_model(frame, conf=0.35, imgsz=640)  # ✅ RAISED from 0.20
        h_boxes = []
        if res_h[0].boxes is not None:
            h_boxes = res_h[0].boxes.xyxy.cpu().numpy()

        # ═════════════════════════════════════════════════════════════
        # STEP 3: TWO-WHEELER DETECTION (HARD-LOCKED)
        # Only YOLO class-3 (motorcycle) detections are used here.
        # This avoids auto/e-rickshaw/car confusion from open-vocabulary labels.
        # ═════════════════════════════════════════════════════════════
        v_boxes = []
        
        # ── Method A: YOLO class 3 (motorcycle) ──
        yolo_res = car_model(frame, classes=[3], conf=0.40)
        if yolo_res[0].boxes is not None:
            for b in yolo_res[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, b)
                w = x2 - x1
                h = y2 - y1
                # Two-wheelers are typically wider than tall in CCTV perspective.
                aspect_wh = w / (h + 1e-5)
                if 0.85 <= aspect_wh <= 4.20 and 900 < w * h < 45000:
                    v_boxes.append([x1, y1, x2, y2])
                    print(f"  [Frame {frame_id}] YOLO bike found: {w}x{h} area={w*h}")

        # ── Method B: fallback world-model for scooters/motorcycles (when class-3 misses) ──
        if not v_boxes:
            for label in ["scooter", "motorcycle", "motorbike"]:
                world_model.set_classes([label])
                res_v = world_model(frame, conf=0.42, imgsz=640)
                if res_v[0].boxes is None:
                    continue
                for b in res_v[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, b)
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    aspect_wh = w / (h + 1e-5)
                    if 0.85 <= aspect_wh <= 4.20 and 900 < area < 45000:
                        v_boxes.append([x1, y1, x2, y2])
                        print(f"  [Frame {frame_id}] World {label} fallback: {w}x{h} area={area}")

        # Remove duplicate two-wheeler boxes
        if v_boxes:
            unique = []
            for b in v_boxes:
                if not any(iou(b, u) > 0.5 for u in unique):
                    unique.append(b)
            v_boxes = unique
            print(f"  [Frame {frame_id}] Total two-wheelers: {len(v_boxes)}")
        else:
            continue  # No two-wheelers detected, skip frame

        # ═════════════════════════════════════════════════════════════
        # STEP 4: MATCH PERSONS TO TWO-WHEELERS & CHECK HELMETS
        # ═════════════════════════════════════════════════════════════
        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            pw = px2 - px1
            ph = py2 - py1
            if pw * ph < AREA_THRESHOLD:
                continue

            # Check person shape (must be seated rider, not standing pedestrian)
            ratio_hw = ph / pw if pw > 0 else 0
            if ratio_hw > 3.5 or ratio_hw < 1.2:  # ✅ RELAXED from 3.0 and 1.5
                print(f"  [Frame {frame_id}] Person rejected: bad aspect ratio {ratio_hw:.2f}")
                continue

            # ─── CHECK 1: HELMET ───
            has_helmet = False
            helmet_overlap = 0
            
            if len(h_boxes) > 0:
                for hbox in h_boxes:
                    hx1, hy1, hx2, hy2 = map(int, hbox)
                    iou_val = iou(pbox, [hx1, hy1, hx2, hy2])
                    helmet_overlap = max(helmet_overlap, iou_val)
                    
                    # Helmet must be on head (top 30% of person)
                    head_region = [px1, py1, px2, py1 + int(ph * 0.30)]
                    iou_head = iou(head_region, [hx1, hy1, hx2, hy2])
                    
                    if iou_head > 0.08 or iou_val > 0.15:  # ✅ RELAXED thresholds
                        has_helmet = True
                        print(f"  [Frame {frame_id}] HELMET DETECTED! overlap={iou_val:.3f}")
                        break
            
            if has_helmet:
                print(f"  [Frame {frame_id}] Person HAS helmet - SKIPPING")
                continue

            # ─── CHECK 2: ON TWO-WHEELER (STRICT) ───
            person_box = [px1, py1, px2, py2]
            on_two_wheeler = False
            matched_vehicle = None

            for vbox in v_boxes:
                # Reuse proven strict rider logic from triple-riding mode.
                if is_person_on_vehicle(person_box, vbox):
                    on_two_wheeler = True
                    matched_vehicle = vbox
                    print(f"  [Frame {frame_id}] ✅ PERSON ON TWO-WHEELER (strict)!")
                    break

            if not on_two_wheeler:
                print(f"  [Frame {frame_id}] Person NOT on two-wheeler - SKIPPING")
                continue

            # ═════════════════════════════════════════════════════════════
            # ✅ MATCH FOUND: RIDER WITHOUT HELMET ON TWO-WHEELER!
            # ═════════════════════════════════════════════════════════════
            label = "🚨 NO HELMET! (Two-Wheeler)"
            key = f"no_helmet_{px1//50}_{py1//50}"
            
            if tracker.should_skip(key, frame_id):
                print(f"  [Frame {frame_id}] Skipped (cooldown)")
                continue
            
            tracker.update(key, frame_id)

            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, "NO HELMET!", (0, 0, 255))  # RED
            if matched_vehicle:
                vx1, vy1, vx2, vy2 = matched_vehicle
                draw_box(ann, vx1, vy1, vx2, vy2, "two-wheeler", (255, 0, 0))  # BLUE

            img_path = os.path.join(SAVE_DIR, f"no_helmet_{get_session_id()}_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 90])

            print(f"  🎯 VIOLATION SAVED! Image: {img_path}\n")

            results_list.append({
                "object": "🚨 Person WITHOUT helmet on two-wheeler",
                "violation": "NO_HELMET",
                "severity": "HIGH",
                "person_bbox": [px1, py1, px2, py2],
                "vehicle_bbox": matched_vehicle if matched_vehicle else [],
                "image_path": img_path,
                "timestamp": frame_id,
                "frame_id": frame_id
            })

    print(f"\n  📊 SUMMARY: {len(results_list)} violations found\n")


# ═════════════════════════════════════════════════════════════════════
# THRESHOLD CHANGES SUMMARY:
# ═════════════════════════════════════════════════════════════════════
# 
# Person detection:      0.45 → 0.35  (catches more people)
# Helmet detection:      0.20 → 0.35  (reduces false positives)  
# YOLO two-wheeler:      0.45 → 0.35  (catches more bikes)
# YOLOWorld two-wheeler: 0.35 → 0.30  (catches more scooters)
#
# Area constraint:       3000-20000 → 1500-30000 (catches all sizes)
# Aspect ratio:          1.5-3.0 → 1.2-3.5 (more flexible)
# IoU threshold:         0.12 → 0.08 (relaxed matching)
# H-overlap threshold:   (varies) → 0.20 (20% horizontal overlap)
# V-overlap threshold:   (varies) → 0.25 (25% vertical overlap)
#
# ═════════════════════════════════════════════════════════════════════

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter + 1e-6
    return inter / union


def unique_persons(persons):
    unique = []
    for p in persons:
        add = True
        for u in unique:
            if compute_iou(p, u) > 0.5:
                add = False
                break
        if add:
            unique.append(p)
    return unique


def is_person_on_vehicle(pbox, vbox):
    """
    STRICT: Sirf vehicle par BAITHA rider count hoga.
    Side mein khade, paas se guzarte, aur adjacent vehicle wale REJECT.
    """
    px1, py1, px2, py2 = pbox
    vx1, vy1, vx2, vy2 = vbox

    pw = px2 - px1
    ph = py2 - py1
    vw = vx2 - vx1
    vh = vy2 - vy1

    if pw <= 0 or ph <= 0 or vw <= 0 or vh <= 0:
        return False

    # ── Rule 1: Standing person reject ─────────────────────────
    # Seated rider aspect: 1.0 to 2.8 | Standing: > 3.2
    aspect = ph / (pw + 1e-5)
    if aspect > 3.2:
        return False

    # ── Rule 2: Person CENTER-X must be INSIDE vehicle (5% max outside)
    # KEY FIX: Pehle 30% tha — isiliye side wale log aa rahe the
    pcx = (px1 + px2) / 2
    if not (vx1 - vw * 0.05 <= pcx <= vx2 + vw * 0.05):
        return False  # person horizontally vehicle ke bahar hai

    # ── Rule 3: Horizontal overlap 30%+ of vehicle width ───────
    horiz_overlap = max(0, min(px2, vx2) - max(px1, vx1))
    horiz_ratio   = horiz_overlap / (vw + 1e-5)
    if horiz_ratio < 0.30:
        return False

    # ── Rule 4: Person bottom vehicle ke TOP se neeche hona chahiye
    p_bottom = py2
    if p_bottom < vy1:
        return False  # vehicle ke upar float — impossible

    # ── Rule 5: CRITICAL — person bottom vehicle bottom se zyada
    # neeche NAHI hona chahiye (10% margin only)
    # KEY FIX: Pehle 25% tha — isiliye road par khade log pass ho rahe the
    # Ground par khada insaan: uska bottom = ground = vehicle bottom
    # Rider: uska bottom footrest par = vehicle bottom ke paas (10% andar)
    if p_bottom > vy2 + vh * 0.10:
        return False

    # ── Rule 6: Person ka lower body (55%) vehicle ke andar hona chahiye
    person_lower_y = py1 + ph * 0.55
    if person_lower_y < vy1:
        return False

    # ── Rule 7: Vertical overlap 40%+ of person height ─────────
    # KEY FIX: Pehle 25% tha — bahut loose
    vert_overlap = max(0, min(py2, vy2) - max(py1, vy1))
    vert_ratio   = vert_overlap / (ph + 1e-5)
    if vert_ratio < 0.40:
        return False

    # ── Rule 8: Person top vehicle top se 1.0x height se zyada upar nahi
    if py1 < vy1 - vh * 1.0:
        return False

    # ── Rule 9: Person width vehicle width se zyada nahi honi chahiye
    # Ek rider vehicle se chowda nahi hoga
    if pw > vw * 0.90:
        return False

    return True


def run_triple_riding_mode(cap, vehicle_type, results_list):
    """
    Triple Riding Detection — STRICT VERSION
   
    Sirf wahi persons count honge jo actually bike/scooter par BAITHE hain.
    Road par khade, paas se guzarte, ya sirf nearby dikhne wale log REJECT honge.
    """
    tracker  = CooldownTracker()
    frame_id = 0

    print(f"  Searching: triple riding on {vehicle_type}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        # ── 1. PERSON DETECTION ────────────────────────────────────
        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.35, imgsz=640)
        if res_p[0].boxes is None:
            continue
        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()

        # ── 2. VEHICLE DETECTION ───────────────────────────────────
        v_boxes = []

        if vehicle_type in ("bike", "two_wheeler"):
            for label in ["bike", "motorcycle", "motorbike"]:
                world_model.set_classes([label])
                res_v = world_model(frame, conf=0.30, imgsz=640)
                if res_v[0].boxes is not None:
                    for b in res_v[0].boxes.xyxy.cpu().numpy():
                        v_boxes.append(list(map(int, b)))
            yolo_res = car_model(frame, classes=[3], conf=0.35)
            if yolo_res[0].boxes is not None:
                for b in yolo_res[0].boxes.xyxy.cpu().numpy():
                    v_boxes.append(list(map(int, b)))

        elif vehicle_type == "scooter":
            for label in ["scooter", "moped"]:
                world_model.set_classes([label])
                res_v = world_model(frame, conf=0.30, imgsz=640)
                if res_v[0].boxes is not None:
                    for b in res_v[0].boxes.xyxy.cpu().numpy():
                        v_boxes.append(list(map(int, b)))
            yolo_res = car_model(frame, classes=[3], conf=0.35)
            if yolo_res[0].boxes is not None:
                for b in yolo_res[0].boxes.xyxy.cpu().numpy():
                    v_boxes.append(list(map(int, b)))

        else:
            yolo_res = car_model(frame, classes=[3], conf=0.30)
            if yolo_res[0].boxes is not None:
                for b in yolo_res[0].boxes.xyxy.cpu().numpy():
                    v_boxes.append(list(map(int, b)))

        # Vehicle boxes deduplicate
        unique_v = []
        for vb in v_boxes:
            if not any(iou(vb, uv) > 0.50 for uv in unique_v):
                unique_v.append(vb)
        v_boxes = unique_v

        if not v_boxes:
            continue

        # ── 3. MATCH PERSONS TO VEHICLE ───────────────────────────
        for vbox in v_boxes:
            vx1, vy1, vx2, vy2 = vbox
            v_area = (vx2 - vx1) * (vy2 - vy1)
            if v_area < 1500:
                continue  # noise — too small

            persons_on_vehicle = []

            for pbox in p_boxes:
                px1, py1, px2, py2 = map(int, pbox)
                if (px2 - px1) * (py2 - py1) < AREA_THRESHOLD:
                    continue

                # ✅ STRICT check — sirf actually baithe hue log
                if is_person_on_vehicle(
                    [px1, py1, px2, py2],
                    [vx1, vy1, vx2, vy2]
                ):
                    persons_on_vehicle.append([px1, py1, px2, py2])

            persons_on_vehicle = unique_persons(persons_on_vehicle)

            print(f"  Frame {frame_id}: vehicle={[vx1,vy1,vx2,vy2]} | "
                  f"riders_on_vehicle={len(persons_on_vehicle)}")

            # ── TRIPLE RIDING TRIGGER ──────────────────────────────
            if len(persons_on_vehicle) >= 3:
                label = f"TRIPLE RIDING! ({len(persons_on_vehicle)} persons)"
                key = f"triple_{vx1//50}_{vy1//50}"
                if tracker.should_skip(key, frame_id):
                    continue
                tracker.update(key, frame_id)

                ann = frame.copy()
                draw_box(ann, vx1, vy1, vx2, vy2,
                         f"{len(persons_on_vehicle)} riders", (255, 0, 255))
                for p in persons_on_vehicle:
                    draw_box(ann, p[0], p[1], p[2], p[3], "rider", (0, 255, 255))

                img_path = os.path.join(SAVE_DIR,
                    f"triple_{get_session_id()}_{frame_id}.jpg")
                cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])

                results_list.append({
                    "object":        label,
                    "person_count":  len(persons_on_vehicle),
                    "vehicle_type":  vehicle_type,
                    "vehicle_bbox":  [vx1, vy1, vx2, vy2],
                    "person_bboxes": persons_on_vehicle,
                    "image_path":    img_path,
                    "timestamp":     frame_id
                })



def run_gender_color_mode(cap, gender_target, color_name, results_list):
    tracker  = CooldownTracker()
    frame_id = 0

    print(f"Searching: {gender_target} wearing {color_name}")

    world_model.set_classes(["person"])

   

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        res = world_model(frame, conf=0.40, imgsz=640)
        if res[0].boxes is None:
            continue

        for box in res[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)

            if (x2 - x1) * (y2 - y1) < AREA_THRESHOLD:
                continue

            # ✅ PERSON CROP (missing tha tere code me)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or is_blurry(crop):
                continue

            # --------------------------------
            # 🎯 FACE DETECTION + GENDER
            # --------------------------------
            face_found = False
            face_crop = None

            try:
                faces = DeepFace.extract_faces(
                    crop,
                    enforce_detection=False
                )

                if faces and len(faces) > 0:
                    face_img = faces[0]["face"]
                    face_crop = (face_img * 255).astype("uint8")
                    face_found = True

            except:
                face_found = False

            # ❌ agar face nahi mila → skip
            if not face_found:
                continue

            # 🎯 Gender detect
            gender = predict_gender_deepface(face_crop)

            if gender != gender_target:
                continue

            # --------------------------------
            # 🎯 COLOR DETECTION (IMPROVED)
            # --------------------------------
            h, w = crop.shape[:2]
            torso = crop[int(h * 0.35):int(h * 0.65),
                         int(w * 0.25):int(w * 0.75)]

            region = torso if torso.size > 0 else crop

           
               

            # ✅ MATCH FOUND
            label = f"{gender_target} wearing {color_name}"

            key = f"gender_{x1//50}_{y1//50}"
            if tracker.should_skip(key, frame_id):
                continue
            tracker.update(key, frame_id)



            target_ratio = color_ratio(region, color_name)
            if target_ratio < 0.35:
                continue
            max_other = 0.0
            for c in COLOR_RANGES:
                if c == color_name: continue
                other_ratio = color_ratio(region, c)
                if other_ratio > max_other: max_other = other_ratio
            if max_other >= target_ratio or target_ratio < max_other * 1.4:
                continue

         

            img_path = save_detection(
                frame,
                [x1, y1, x2, y2],
                label,
                frame_id,
                "gender",
                (255, 0, 255)
            )

            results_list.append({
                "object": label,
                "gender": gender,
                "color": color_name,
                "bbox": [x1, y1, x2, y2],
                "image_path": img_path,
                "timestamp": frame_id
            })
def run_person_helmet_color_mode(cap, color_name, results_list):
    tracker = CooldownTracker()
    frame_id = 0
    color_name = normalize_color_name(color_name)
    print(f"Searching: person with {color_name} helmet")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.40, imgsz=640)
        if res_p[0].boxes is None: continue

        world_model.set_classes(["helmet"])
        res_h = world_model(frame, conf=0.35, imgsz=640)
        if res_h[0].boxes is None: continue

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
        h_boxes = res_h[0].boxes.xyxy.cpu().numpy()

        # Best match per helmet (avoid duplicate)
        used_helmets = set()
        for h_idx, hbox in enumerate(h_boxes):
            hx1, hy1, hx2, hy2 = map(int, hbox)
            # Strict helmet-person check
            best_p_idx = -1
            best_iou = 0.0
            for p_idx, pbox in enumerate(p_boxes):
                if helmet_belongs_to_person_strict(pbox, hbox):  # use strict function
                    iou_val = iou(pbox, hbox)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_p_idx = p_idx
            if best_p_idx == -1:
                continue

            pbox = p_boxes[best_p_idx]
            px1, py1, px2, py2 = map(int, pbox)
            if (px2-px1)*(py2-py1) < AREA_THRESHOLD:
                continue

            # Helmet crop
            helmet_crop = frame[hy1:hy2, hx1:hx2]
            if helmet_crop.size == 0 or is_blurry(helmet_crop):
                continue

            # 🎯 Color check (calibrated for small helmet crops)
            target_ratio = color_ratio(helmet_crop, color_name)
            base_thr = get_color_threshold(color_name)
            # Helmet is a small object -> use practical threshold instead of fixed 0.35
            helmet_thr = max(0.10, base_thr * 1.20)
            if target_ratio < helmet_thr:
                continue

            # Competitor color check
            max_other = 0.0
            for c in COLOR_RANGES:
                if c == color_name: continue
                other_ratio = color_ratio(helmet_crop, c)
                if other_ratio > max_other:
                    max_other = other_ratio
            neutral_colors = {"black", "white", "grey", "gray", "silver"}
            if color_name in neutral_colors:
                # Neutrals are confusion-prone; require clearer separation
                if max_other > target_ratio * 1.02:
                    continue
            else:
                # Vivid colors: allow close competition due to reflections/stickers
                if max_other > target_ratio * 1.28:
                    continue

            # ✅ MATCH FOUND
            label = f"person with {color_name} helmet"
            key = f"helmet_color_{px1//50}_{py1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, "person", (0,255,0))
            draw_box(ann, hx1, hy1, hx2, hy2, f"{color_name} helmet", (255,0,0))
            img_path = os.path.join(SAVE_DIR, f"helmet_color_{get_session_id()}_{frame_id}.jpg")
            cv2.imwrite(img_path, ann)

            results_list.append({
                "object": label,
                "color": color_name,
                "person_bbox": [px1, py1, px2, py2],
                "helmet_bbox": [hx1, hy1, hx2, hy2],
                "image_path": img_path,
                "timestamp": frame_id
            })





def run_person_helmet_any_mode(cap, prompt, results_list):
    tracker = CooldownTracker()
    frame_id = 0
    two_wheeler_info = {"strategy": "yolo_class", "class_ids": [3]}  # class 3 = motorcycle

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0: continue

        # Detect persons
        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.40, imgsz=640)
        if res_p[0].boxes is None: continue

        # Detect helmets
        world_model.set_classes(["helmet"])
        res_h = world_model(frame, conf=0.35, imgsz=640)
        if res_h[0].boxes is None: continue

        # Detect two-wheelers
        v_boxes = detect_vehicles_in_frame(frame, two_wheeler_info, conf_yolo=0.25)

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()
        h_boxes = res_h[0].boxes.xyxy.cpu().numpy()

        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            if (px2-px1)*(py2-py1) < AREA_THRESHOLD: continue

            # Check if person has helmet (strict spatial)
            has_helmet = False
            for hbox in h_boxes:
                hx1, hy1, hx2, hy2 = map(int, hbox)
                if helmet_belongs_to_person_strict([px1, py1, px2, py2], [hx1, hy1, hx2, hy2]):
                    has_helmet = True
                    break
            if not has_helmet:
                continue

            # ----- NEW: Check if person is on a two-wheeler -----
            person_center_x = (px1 + px2) / 2
            person_bottom_y = py2
            person_width = px2 - px1

            on_two_wheeler = False
            for vbox in v_boxes:
                vx1, vy1, vx2, vy2 = vbox
                v_width = vx2 - vx1
                # Horizontal overlap
                overlap = max(0, min(px2, vx2) - max(px1, vx1))
                overlap_ratio = overlap / v_width if v_width > 0 else 0
                vertical_gap = abs(person_bottom_y - vy1)
                if overlap_ratio > 0.15 and vertical_gap < person_width * 1.2:
                    on_two_wheeler = True
                    break
                # Distance check
                dist = np.sqrt(((person_center_x - (vx1+vx2)/2)**2 + (person_bottom_y - vy1)**2))
                if dist < person_width * 1.5:
                    on_two_wheeler = True
                    break

            if not on_two_wheeler:
                continue  # person not on two-wheeler (e.g., in car), skip

            # Save detection
            label = "person with helmet (on two-wheeler)"
            key = f"helmet_any_{px1//50}_{py1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, "person", (0,255,0))
            draw_box(ann, hx1, hy1, hx2, hy2, "helmet", (255,0,0))
            # optionally draw two-wheeler box
            img_path = os.path.join(SAVE_DIR, f"helmet_any_{get_session_id()}_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
            results_list.append({
                "object": label,
                "person_bbox": [px1, py1, px2, py2],
                "helmet_bbox": [hx1, hy1, hx2, hy2],
                "image_path": img_path,
                "timestamp": frame_id
            })
# ================================
# VEHICLE DETECTION HELPER
# ================================

def detect_vehicles_in_frame(frame, vehicle_info, conf_yolo=0.40, conf_world=0.38):
    boxes = []

    def _is_three_wheeler_like(box):
        x1, y1, x2, y2 = box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h
        aspect = w / (h + 1e-6)
        # Three-wheelers are usually compact; long-thin boxes are often bikes.
        if area < 1200:
            return False
        if aspect < 0.75 or aspect > 2.80:
            return False
        return True

    if vehicle_info["strategy"] == "yolo_class":
        res = car_model(frame, classes=vehicle_info["class_ids"], conf=conf_yolo)
        if res[0].boxes is not None:
            for b in res[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, b)
                w, h = x2-x1, y2-y1
                if vehicle_info["class_ids"] == [2]:
                    if h / (w + 1e-5) > 2.5 or w / (h + 1e-5) > 4.5:
                        continue
                if w * h < 800:
                    continue
                boxes.append([x1, y1, x2, y2])
    else:
        wl = vehicle_info["world_label"].lower()
        if wl in ("scooter", "moped"):
            conf_world = 0.28
        if wl == "electric rickshaw":
            conf_world = min(conf_world, 0.30)
        world_queries = [vehicle_info["world_label"]]
        allowed_labels = {wl}
        if wl == "electric rickshaw":
            # Real-world model outputs vary; accept close variants for e-rickshaw intent.
            world_queries = ["electric rickshaw", "e-rickshaw", "auto rickshaw", "rickshaw"]
            allowed_labels = {"electric rickshaw", "e rickshaw", "auto rickshaw", "rickshaw"}

        for query_label in world_queries:
            world_model.set_classes([query_label])
            res = world_model(frame, conf=conf_world, imgsz=640)
            if res[0].boxes is None:
                continue
            for box, cls_id in zip(res[0].boxes.xyxy.cpu().numpy(),
                                   res[0].boxes.cls.cpu().numpy()):
                pred_label = world_model.names[int(cls_id)].lower().replace("-", " ").strip()
                pred_label = re.sub(r"\s+", " ", pred_label)
                if pred_label not in allowed_labels:
                    continue
                x1, y1, x2, y2 = map(int, box)
                if (x2-x1) * (y2-y1) < 500:
                    continue
                boxes.append([x1, y1, x2, y2])

        # Guardrail for e-rickshaw/auto-rickshaw prompts:
        # YOLOWorld can sometimes map bikes as "auto rickshaw" in crowded scenes.
        # Remove candidate auto-rickshaw boxes that strongly overlap two-wheeler boxes.
        if wl in ("auto rickshaw", "electric rickshaw") and boxes:
            boxes = [b for b in boxes if _is_three_wheeler_like(b)]
            two_wheeler_boxes = []

            # Strong two-wheeler negatives from YOLO class-3.
            tw_yolo = car_model(frame, classes=[3], conf=0.30)
            if tw_yolo[0].boxes is not None:
                for tw_box in tw_yolo[0].boxes.xyxy.cpu().numpy():
                    tx1, ty1, tx2, ty2 = map(int, tw_box)
                    if (tx2 - tx1) * (ty2 - ty1) < 450:
                        continue
                    two_wheeler_boxes.append([tx1, ty1, tx2, ty2])

            for tw_label in ("bike", "motorcycle", "motorbike", "scooter"):
                world_model.set_classes([tw_label])
                tw_res = world_model(frame, conf=max(conf_world, 0.38), imgsz=640)
                if tw_res[0].boxes is None:
                    continue
                for tw_box in tw_res[0].boxes.xyxy.cpu().numpy():
                    tx1, ty1, tx2, ty2 = map(int, tw_box)
                    if (tx2 - tx1) * (ty2 - ty1) < 450:
                        continue
                    two_wheeler_boxes.append([tx1, ty1, tx2, ty2])

            if two_wheeler_boxes:
                boxes = [
                    b for b in boxes
                    if not any(iou(b, twb) > 0.20 for twb in two_wheeler_boxes)
                ]

    # NMS dedup
    unique = []
    for b in boxes:
        if not any(iou(b, u) > 0.45 for u in unique):
            unique.append(b)
    return unique


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
    px1, py1, px2, py2 = p_box
    hx1, hy1, hx2, hy2 = h_box
   
    ph = py2 - py1
    pw = px2 - px1
   
    # Helmet ka center
    hcx = (hx1 + hx2) / 2
    hcy = (hy1 + hy2) / 2
   
    # 1. Helmet person ke upper 35% area mein hona chahiye
    if not (px1 <= hcx <= px2 and hcy <= py1 + ph * 0.30):
        return False
   
    # 2. Helmet ka bottom person ke shoulder se upar hona chahiye (kyuki helmet sir par hota hai)
    if hy2 > py1 + ph * 0.45:   # 45% se neeche gaya toh galat
        return False
   
    # 3. Helmet person ke bahar bahut jyada nahi hona chahiye (side se)
    if hx1 < px1 - pw * 0.2 or hx2 > px2 + pw * 0.2:
        return False
   
    # 4. IoU threshold increase karo (0.04 se 0.10)
    iou_val = iou(p_box, h_box)
    if iou_val < 0.15:
        return False
   
    # 5. Helmet ka width person ke width ka 30% se 70% ke beech hona chahiye
    hw = hx2 - hx1
    if not (0.30 * pw <= hw <= 0.75 * pw):
        return False
   
    return True


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
    tracker  = CooldownTracker(cooldown=40)
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
        direct_results = plate_model(frame, conf=0.20, imgsz=640)
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

            img_path = os.path.join(SAVE_DIR, f"plate_{get_session_id()}_{frame_id}.jpg")
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




# ---- 2. COLOR + VEHICLE ---- (COMPLETE FIX: Ultra-Strict Color Matching)

def run_color_object_mode(cap, color_name, vehicle_word, vehicle_info, results_list):
    """
    Accurate color + vehicle detection.
    
    Improvements over old version:
    1. Multi-zone body sampling (strips road/sky background)
    2. Per-color adaptive thresholds
    3. Confusion-pair exclusion
    4. Temporal vote smoothing (3/5 frames must agree)
    5. Proper NMS dedup of overlapping vehicle boxes
    6. Saves color confidence score in result
    """
    cooldown_tracker = CooldownTracker()
    vehicle_tracker  = VehicleTracker(required_votes=2, window=5)
    frame_id         = 0
    detected_count   = 0

    print(f"\n{'='*60}")
    print(f"  Searching: {color_name.upper() if color_name else 'ANY COLOR'} {vehicle_word.upper()}")
    print(f"  Strategy:  {vehicle_info['strategy']}")
    print(f"  Threshold: {VEHICLE_COLOR_THRESHOLDS.get(color_name, 0.20) if color_name else 'N/A'}")
    print(f"{'='*60}")

    is_car_like_query = (
        vehicle_info.get("strategy") == "yolo_class" and
        2 in vehicle_info.get("class_ids", [])
    )

    if is_car_like_query:
        # Used to re-validate ambiguous "car" candidates and reject boards/signage.
        world_model.set_classes(["car", "vehicle", "automobile", "sedan", "hatchback", "suv"])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        # ── 1. DETECT VEHICLES (dual strategy) ──────────────────
        raw_boxes = []

        if vehicle_info["strategy"] == "yolo_class":
            # Slightly higher conf removes many signboard/background false positives
            yolo_conf = 0.35 if is_car_like_query else 0.38
            res = car_model(frame, classes=vehicle_info["class_ids"], conf=yolo_conf)
            if res[0].boxes is not None:
                for b, bconf in zip(
                    res[0].boxes.xyxy.cpu().numpy(),
                    res[0].boxes.conf.cpu().numpy()
                ):
                    x1, y1, x2, y2 = map(int, b)
                    w, h = x2-x1, y2-y1
                    aspect = w / (h + 1e-5)
                    # Basic sanity: not a sliver
                    if w < 30 or h < 20:
                        continue
                    # For cars: not an impossibly tall/thin box
                    if vehicle_info.get("class_ids") == [2]:
                        # Board-like or pole-like shapes often show as false "car"
                        if aspect < 0.75 or aspect > 3.2:
                            continue
                        # Very small/flat boxes are frequently text boards
                        if h < 30 or w < 45:
                            continue
                        # Extra confidence gate for car class
                        if float(bconf) < 0.35:
                            continue
                    raw_boxes.append({
                        "box": [x1, y1, x2, y2],
                        "det_conf": float(bconf),
                        "source": "yolo_class"
                    })

        else:  # YOLOWorld
            world_model.set_classes([vehicle_info["world_label"]])
            conf_world = 0.28 if vehicle_info["world_label"].lower() in ("scooter", "moped") else 0.35
            res = world_model(frame, conf=conf_world, imgsz=640)
            if res[0].boxes is not None:
                for box, cls_id, conf_val in zip(
                    res[0].boxes.xyxy.cpu().numpy(),
                    res[0].boxes.cls.cpu().numpy(),
                    res[0].boxes.conf.cpu().numpy()
                ):
                    pred_label = world_model.names[int(cls_id)].lower()
                    if pred_label != vehicle_info["world_label"].lower():
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    if (x2-x1) * (y2-y1) < 1500:
                        continue
                    raw_boxes.append({
                        "box": [x1, y1, x2, y2],
                        "det_conf": float(conf_val),
                        "source": "world"
                    })

            # Also run YOLOv8n class-based as backup (catches what world misses)
            backup_classes = []
            wl = vehicle_info["world_label"].lower()
            if wl in ("motorcycle", "motorbike", "bike", "scooter", "moped"):
                backup_classes = [3]
            elif wl in ("car",):
                backup_classes = [2]
            elif wl in ("bus",):
                backup_classes = [5]
            elif wl in ("truck", "lorry"):
                backup_classes = [7]

            if backup_classes:
                res2 = car_model(frame, classes=backup_classes, conf=0.35)
                if res2[0].boxes is not None:
                    for b, bconf in zip(
                        res2[0].boxes.xyxy.cpu().numpy(),
                        res2[0].boxes.conf.cpu().numpy()
                    ):
                        raw_boxes.append({
                            "box": list(map(int, b)),
                            "det_conf": float(bconf),
                            "source": "yolo_backup"
                        })

        if not raw_boxes:
            continue

        # ── 2. NMS DEDUP ────────────────────────────────────────
        unique_boxes = []
        for item in raw_boxes:
            b = item["box"]
            merged = False
            for i, u in enumerate(unique_boxes):
                if iou(b, u["box"]) > 0.45:
                    # Keep higher-confidence detection for overlapping boxes
                    if item["det_conf"] > u["det_conf"]:
                        unique_boxes[i] = item
                    merged = True
                    break
            if not merged:
                unique_boxes.append(item)

        # ── 3. QUALITY GATE + COLOR CHECK ───────────────────────
        for item in unique_boxes:
            box = item["box"]
            det_conf = item["det_conf"]
            x1, y1, x2, y2 = box
            area = (x2-x1) * (y2-y1)
            frame_area = frame.shape[0] * frame.shape[1]

            if area < AREA_THRESHOLD:
                continue
            if is_car_like_query and area < max(1800, int(frame_area * 0.0022)):
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or is_blurry(crop):
                continue

            # Skip if vehicle is clipped at frame edge (unreliable color)
            frame_h, frame_w = frame.shape[:2]
            edge_clip = (x1 < 5 or y1 < 5 or
                         x2 > frame_w - 5 or y2 > frame_h - 5)
            if edge_clip and area < 8000:
                # Only skip small clipped boxes; large ones are usually fine
                continue

            # Car-specific re-check: reject board/sign-like false positives
            if is_car_like_query:
                ex = int((x2 - x1) * 0.15)
                ey = int((y2 - y1) * 0.15)
                rx1 = max(0, x1 - ex); ry1 = max(0, y1 - ey)
                rx2 = min(frame.shape[1], x2 + ex); ry2 = min(frame.shape[0], y2 + ey)
                re_crop = frame[ry1:ry2, rx1:rx2]
                if re_crop.size == 0:
                    continue
                wres = world_model(re_crop, conf=0.28, imgsz=512)
                if wres[0].boxes is None or len(wres[0].boxes) == 0:
                    continue
                # Require at least one substantial world-box in the re-crop.
                rw = rx2 - rx1; rh = ry2 - ry1
                min_rel_area = 0.12 * rw * rh
                valid_world_box = False
                for wb in wres[0].boxes.xyxy.cpu().numpy():
                    wx1, wy1, wx2, wy2 = map(int, wb)
                    if (wx2 - wx1) * (wy2 - wy1) >= min_rel_area:
                        valid_world_box = True
                        break
                if not valid_world_box:
                    continue

            # ── COLOR CHECK (multi-zone) ──────────────────────
            color_ok, confidence, details = vehicle_color_match(frame, box, color_name)

            # If strict color check fails, run relaxed fallback for car-like queries.
            # This catches valid red cars under shade/low-light where multi-zone vote is too strict.
            neutral_colors = {"black", "white", "grey", "gray", "silver"}
            if (
                (not color_ok)
                and color_name
                and is_car_like_query
                and normalize_color_name(color_name) not in neutral_colors
            ):
                norm_color = normalize_color_name(color_name)
                h, w = crop.shape[:2]
                relaxed_crop = crop[int(h * 0.20):int(h * 0.85), int(w * 0.08):int(w * 0.92)]
                if relaxed_crop.size > 0:
                    relaxed_ratio = color_ratio(relaxed_crop, norm_color)
                    relax_mult = get_color_calibration(norm_color).get("relaxed_mult", 0.62)
                    relaxed_threshold = max(0.10, VEHICLE_COLOR_THRESHOLDS.get(norm_color, 0.20) * relax_mult)
                    max_other_relaxed = max(
                        (color_ratio(relaxed_crop, c) for c in COLOR_RANGES if c != norm_color),
                        default=0.0
                    )
                    if relaxed_ratio >= relaxed_threshold and relaxed_ratio >= (max_other_relaxed * 0.85):
                        color_ok = True
                        confidence = round(relaxed_ratio, 3)
                        details = {
                            "zones": details.get("zones", []),
                            "passed": details.get("passed", 0),
                            "relaxed_fallback": True,
                            "relaxed_ratio": round(relaxed_ratio, 3),
                            "relaxed_threshold": round(relaxed_threshold, 3),
                        }

            # Hard lock: return only exact requested color when strict mode is enabled.
            if color_ok and color_name and STRICT_PROMPT_MATCH:
                norm_color = normalize_color_name(color_name)
                h, w = crop.shape[:2]
                core = crop[int(h * 0.20):int(h * 0.82), int(w * 0.10):int(w * 0.90)]
                if core.size == 0:
                    continue
                dom_color = normalize_color_name(get_dominant_color_name(core))
                target_r = color_ratio(core, norm_color)
                dom_r = color_ratio(core, dom_color) if dom_color in COLOR_RANGES else 0.0
                red_r = color_ratio(core, "red")
                black_r = color_ratio(core, "black")
                white_r = color_ratio(core, "white")
                grey_r = max(color_ratio(core, "grey"), color_ratio(core, "gray"))
                silver_r = color_ratio(core, "silver")
                max_other = max(
                    (color_ratio(core, c) for c in COLOR_RANGES if c != norm_color),
                    default=0.0
                )

                # Requested color must be dominant (or extremely close) in core body region.
                if dom_color != norm_color and target_r < (dom_r * 0.97):
                    continue

                vivid_colors = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "olive"}
                color_cfg = get_color_calibration(norm_color)
                # Hard veto rules to prevent cross-color confusion while keeping vivid colors usable.
                if norm_color == "black":
                    if target_r < color_cfg.get("min_ratio", 0.16):
                        continue
                    if target_r < red_r * 1.30 or target_r < white_r * 1.20 or target_r < silver_r * 1.20:
                        continue
                elif norm_color == "white":
                    if target_r < color_cfg.get("min_ratio", 0.14):
                        continue
                    if target_r < black_r * 1.18 or target_r < red_r * 1.12:
                        continue
                elif norm_color in ("grey", "gray", "silver"):
                    neutral_peak = max(grey_r, silver_r, white_r, black_r)
                    if target_r < color_cfg.get("min_ratio", 0.12) or target_r < neutral_peak * 0.92:
                        continue
                elif norm_color in vivid_colors:
                    # Vivid colors can appear with black windows/tyres/shadows.
                    # Avoid over-strict dominance so true red/blue/green cars are not dropped.
                    if target_r < color_cfg.get("min_ratio", 0.10):
                        continue
                    if target_r < max_other * color_cfg.get("dominance_mult", 0.78):
                        continue
                else:
                    if target_r < color_cfg.get("min_ratio", 0.10) or target_r < max_other * color_cfg.get("dominance_mult", 0.90):
                        continue

            # Update temporal tracker
            confirmed = vehicle_tracker.update(box, frame_id, color_ok)

            if color_ok:
                print(f"    Frame {frame_id}: color={color_name} conf={confidence:.2f} "
                      f"zones_passed={details.get('passed', '?')} area={area}")
            
            if not confirmed:
                continue   # waiting for more frame votes

            # ── DEDUP + SAVE ──────────────────────────────────
            cx = (x1 + x2) // 12
            cy = (y1 + y2) // 12
            key = f"{color_name}_{vehicle_word}_{cx}_{cy}"

            if cooldown_tracker.should_skip(key, frame_id):
                continue
            cooldown_tracker.update(key, frame_id)

            detected_count += 1

            # Final confidence must reflect both: "is this a vehicle?" + "is color correct?"
            final_conf = (0.55 * float(det_conf)) + (0.45 * float(confidence))
            if details.get("relaxed_fallback"):
                final_conf *= 0.82
            final_conf = max(0.0, min(0.99, final_conf))

            if color_name:
                label = f"{color_name} {vehicle_word} ({int(final_conf*100)}%)"
            else:
                dom = get_dominant_color_name(
                    frame[y1:y2, x1:x2][
                        int((y2-y1)*0.15):int((y2-y1)*0.72),
                        int((x2-x1)*0.08):int((x2-x1)*0.92)
                    ]
                )
                label = f"{dom} {vehicle_word}"

            print(f"  ✅ SAVED #{detected_count} | frame={frame_id} | {label}")

            results_list.append({
                "object":        label,
                "color":         color_name or "any",
                "vehicle":       vehicle_word,
                "confidence":    final_conf,
                "vehicle_confidence": float(det_conf),
                "color_confidence": float(confidence),
                "bbox":          [x1, y1, x2, y2],
                "image_path":    save_detection(
                    frame, [x1, y1, x2, y2],
                    label, frame_id, "color_veh"),
                "timestamp":     frame_id,
                "zone_details":  details
            })

        # Periodic tracker cleanup
        if frame_id % 60 == 0:
            vehicle_tracker.cleanup(frame_id)

    print(f"\n  Total {color_name} {vehicle_word} detected: {detected_count}")
    print(f"{'='*60}\n")


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



            required = set(attributes)
            found_attrs = set()
            for a_box, attr in attr_det:
                if attr_belongs_to_person([px1,py1,px2,py2], a_box, attr):
                    found_attrs.add(attr)
            if required.issubset(found_attrs):
                label = "person with " + " & ".join(required)

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

            if not detect_color_in_crop(color_region, color_name, threshold=0.35):
                continue

            # ✅ ALL CONDITIONS MET
            label = f"person with {attribute} wearing {color_name}"
            key = f"attr_color_{px1//50}_{py1//50}"
            if tracker.should_skip(key, frame_id): continue
            tracker.update(key, frame_id)

            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, label, (0, 200, 255))
            img_path = os.path.join(SAVE_DIR, f"attr_color_{get_session_id()}_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])

            results_list.append({
                "object": label,
                "attribute": attribute,
                "color": color_name,
                "image_path": img_path,
                "bbox": [px1, py1, px2, py2],
                "timestamp": frame_id
            })

def predict_gender_deepface(crop):
    try:
        result = DeepFace.analyze(
            crop,
            actions=['gender'],
            enforce_detection=False
        )
       
        # DeepFace kabhi list return karta hai
        if isinstance(result, list):
            result = result[0]

        gender = result.get("dominant_gender", "").lower()

        if gender in ["man", "male"]:
            return "male"
        elif gender in ["woman", "female"]:
            return "female"
       


        return "unknown"

    except Exception as e:
        return "unknown"


# ---- 3.6 NEW: PERSON + VEHICLE + COLOR ----

def run_person_vehicle_color_mode(cap, vehicle_word, vehicle_info, color_name, results_list):
    tracker  = CooldownTracker()
    frame_id = 0

    print(f"  Searching: person on {vehicle_word} wearing {color_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.35, imgsz=640)
        if res_p[0].boxes is None or len(res_p[0].boxes) == 0:
            continue
        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()

        v_boxes = detect_vehicles_in_frame(frame, vehicle_info)
        if not v_boxes:
            continue

        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)
            if (px2 - px1) * (py2 - py1) < AREA_THRESHOLD:
                continue
            person_crop = frame[py1:py2, px1:px2]
            if person_crop.size == 0 or is_blurry(person_crop):
                continue

            # Vehicle proximity check
            person_close = False
            vehicle_bbox = None
            prox = max(px2 - px1, py2 - py1) * 0.8

            for vbox in v_boxes:
                vx1, vy1, vx2, vy2 = vbox
                dist = np.sqrt(
                    ((px1 + px2) / 2 - (vx1 + vx2) / 2) ** 2 +
                    ((py1 + py2) / 2 - (vy1 + vy2) / 2) ** 2
                )
                if dist <= prox:
                    person_close = True
                    vehicle_bbox = vbox
                    break

            if not person_close or vehicle_bbox is None:
                continue

            # ✅ COLOR CHECK PEHLE — save baad mein
            h, w = person_crop.shape[:2]
            torso = person_crop[int(h * 0.25):int(h * 0.75),
                                int(w * 0.15):int(w * 0.85)]
            color_region = torso if torso.size > 0 else person_crop

            target_ratio = color_ratio(color_region, color_name)
            if target_ratio < 0.30:
                continue

            max_other = 0.0
            for c in COLOR_RANGES:
                if c == color_name:
                    continue
                other_ratio = color_ratio(color_region, c)
                if other_ratio > max_other:
                    max_other = other_ratio
            if max_other >= target_ratio * 1.4:
                continue

            # ✅ MATCH — ab save karo
            key = f"veh_color_{px1 // 50}_{py1 // 50}"
            if tracker.should_skip(key, frame_id):
                continue
            tracker.update(key, frame_id)

            label = f"person on {vehicle_word} wearing {color_name}"
            ann = frame.copy()
            draw_box(ann, px1, py1, px2, py2, "person", (0, 255, 0))
            draw_box(ann, vehicle_bbox[0], vehicle_bbox[1],
                     vehicle_bbox[2], vehicle_bbox[3], vehicle_word, (255, 100, 0))
            cv2.putText(ann, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            img_path = os.path.join(SAVE_DIR, f"veh_color_{get_session_id()}_{frame_id}.jpg")
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 85])

            results_list.append({
                "object":       label,
                "vehicle":      vehicle_word,
                "color":        color_name,
                "color_ratio":  round(target_ratio, 3),
                "person_bbox":  [px1, py1, px2, py2],
                "vehicle_bbox": vehicle_bbox,
                "image_path":   img_path,
                "timestamp":    frame_id
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

            prox = max(px2-px1, py2-py1) * 0.8
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
                path = os.path.join(SAVE_DIR, f"pv_{get_session_id()}_{frame_id}.jpg")
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









# ================================
# REFERENCE FACE EMBEDDING EXTRACT
# ================================

def extract_face_embedding(image_bgr):
    """
    BGR image se face embedding extract karo using DeepFace.
    Returns embedding list ya None agar face nahi mila.
    """
    try:
        result = DeepFace.represent(
            image_bgr,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend="opencv"
        )
        if result and len(result) > 0:
            return np.array(result[0]["embedding"])
    except Exception as e:
        print(f"  Face embedding failed: {e}")
    return None


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def verify_face_match(ref_embedding, crop_bgr, threshold=0.78):
    """
    crop_bgr me se face nikalo aur ref_embedding se compare karo.
    Returns (matched: bool, score: float)
    """
    try:
        result = DeepFace.represent(
            crop_bgr,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend="opencv"
        )
        if not result or len(result) == 0:
            return False, 0.0

        crop_emb = np.array(result[0]["embedding"])
        score = cosine_similarity(ref_embedding, crop_emb)
        return score >= threshold, float(score)

    except Exception as e:
        return False, 0.0



# ================================
# PERSON APPEARANCE FUNCTIONS
# ================================

def extract_person_appearance(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None

    h, w = img_bgr.shape[:2]
    if h < 20 or w < 10:
        return None

    if h < 100:
        scale = 100 / h
        img_bgr = cv2.resize(img_bgr,
                             (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        h, w = img_bgr.shape[:2]

    upper = img_bgr[int(h*0.10):int(h*0.50), int(w*0.10):int(w*0.90)]
    lower = img_bgr[int(h*0.50):int(h*0.90), int(w*0.10):int(w*0.90)]
    full  = img_bgr[int(h*0.05):int(h*0.95), :]

    features = {}

    for name, region in [("upper", upper), ("lower", lower), ("full", full)]:
        if region.size == 0:
            continue
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        h_hist = cv2.calcHist([hsv], [0], None, [36], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])

        cv2.normalize(h_hist, h_hist)
        cv2.normalize(s_hist, s_hist)
        cv2.normalize(v_hist, v_hist)

        features[f"{name}_h"] = h_hist.flatten()
        features[f"{name}_s"] = s_hist.flatten()
        features[f"{name}_v"] = v_hist.flatten()
        features[f"{name}_color"] = get_dominant_color_name(region)

    try:
        gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (32, 64))
        hog = cv2.HOGDescriptor((32, 64), (16, 16), (8, 8), (8, 8), 9)
        hog_feat = hog.compute(gray_resized)
        features["hog"] = hog_feat.flatten()
    except Exception:
        features["hog"] = None

    return features


def compare_appearances(ref_feat, query_feat):
    if ref_feat is None or query_feat is None:
        return 0.0

    # NOTE: upper-color "hard gate" causes many false negatives in CCTV due to
    # lighting/white-balance changes. We soft-penalize instead.
    ref_upper = ref_feat.get("upper_color", "unknown")
    q_upper   = query_feat.get("upper_color", "unknown")

    if (ref_upper != "unknown" and
            q_upper != "unknown" and
            ref_upper != q_upper):
        # soft penalty (still allows match if other features are strong)
        color_mismatch_penalty = 0.25
    else:
        color_mismatch_penalty = 1.0

    scores  = []
    weights = []

    # Upper body histogram — highest weight
    for key in ["upper_h", "upper_s"]:
        if key in ref_feat and key in query_feat:
            try:
                dist = cv2.compareHist(
                    ref_feat[key].reshape(-1, 1),
                    query_feat[key].reshape(-1, 1),
                    cv2.HISTCMP_BHATTACHARYYA
                )
                scores.append(max(0.0, 1.0 - dist))
                weights.append(3.0)
            except Exception:
                pass

    # Lower body histogram
    for key in ["lower_h", "lower_s"]:
        if key in ref_feat and key in query_feat:
            try:
                dist = cv2.compareHist(
                    ref_feat[key].reshape(-1, 1),
                    query_feat[key].reshape(-1, 1),
                    cv2.HISTCMP_BHATTACHARYYA
                )
                scores.append(max(0.0, 1.0 - dist))
                weights.append(2.0)
            except Exception:
                pass

    # Full body histogram
    for key in ["full_h", "full_s", "full_v"]:
        if key in ref_feat and key in query_feat:
            try:
                dist = cv2.compareHist(
                    ref_feat[key].reshape(-1, 1),
                    query_feat[key].reshape(-1, 1),
                    cv2.HISTCMP_BHATTACHARYYA
                )
                scores.append(max(0.0, 1.0 - dist))
                weights.append(1.0)
            except Exception:
                pass

    # Lower color match bonus
    ref_lower = ref_feat.get("lower_color", "unknown")
    q_lower   = query_feat.get("lower_color", "unknown")
    lower_match = (ref_lower == q_lower or
                   ref_lower == "unknown" or
                   q_lower   == "unknown")
    scores.append(0.75 if lower_match else 0.0)
    weights.append(1.5)

    # HOG similarity
    if ref_feat.get("hog") is not None and query_feat.get("hog") is not None:
        try:
            rh  = ref_feat["hog"]
            qh  = query_feat["hog"]
            sim = float(np.dot(rh, qh) / (
                np.linalg.norm(rh) * np.linalg.norm(qh) + 1e-8))
            scores.append(max(0.0, sim))
            weights.append(0.5)
        except Exception:
            pass

    if not scores:
        return 0.0

    total_w = sum(weights)
    base = float(sum(s * w for s, w in zip(scores, weights)) / total_w)
    return float(base * color_mismatch_penalty)




# ---- 5.5. REFERENCE IMAGE SEARCH MODE (FACE-BASED PERSON SEARCH) ----


def run_reference_image_search_mode(cap, ref_feat, ref_image, results_list):
    """
    DUAL-MODE Person Search:
    1. Face Recognition (Facenet512) — primary
    2. Appearance Re-ID (clothing + shape) — secondary
   
    Scoring:
    - Face match (high conf) + Appearance match → VERY HIGH confidence
    - Face match only → HIGH confidence  
    - Appearance match only → MEDIUM confidence (strict threshold)
    - Neither → REJECT
    """
    tracker       = CooldownTracker(cooldown=20)
    frame_id      = 0
    match_count   = 0
    reject_count  = 0
    top_candidates = []  # store best near-misses to avoid empty output

    print(f"\n{'='*60}")
    print(f"  🔍 DUAL-MODE PERSON SEARCH")
    print(f"  Mode: Face Recognition + Appearance Re-ID")

    # ══════════════════════════════════════════════════════════
    # STEP 1: Reference features extract karo (BOTH methods)
    # ══════════════════════════════════════════════════════════

    # ── 1A: Face embedding ────────────────────────────────────
    ref_face_embedding = None
    ref_face_conf      = 0.0
    face_mode_available = False

    try:
        backends = ["retinaface", "mtcnn", "opencv", "ssd"]
        best_emb  = None
        best_conf = 0.0

        for backend in backends:
            try:
                faces = DeepFace.extract_faces(
                    ref_image,
                    enforce_detection=False,
                    detector_backend=backend,
                    align=True
                )
                if not faces:
                    continue
                for face_obj in faces:
                    fc = face_obj.get("confidence", 0.0)
                    if fc > best_conf and fc >= 0.50:
                        face_arr = face_obj.get("face")
                        if face_arr is not None:
                            face_bgr = cv2.cvtColor(
                                (face_arr * 255).astype(np.uint8),
                                cv2.COLOR_RGB2BGR
                            )
                            fh, fw = face_bgr.shape[:2]
                            if fh >= 15 and fw >= 15:
                                # Upscale + enhance small faces
                                if min(fh, fw) < 80:
                                    sc = 80 / min(fh, fw)
                                    face_bgr = cv2.resize(
                                        face_bgr,
                                        (int(fw*sc), int(fh*sc)),
                                        interpolation=cv2.INTER_CUBIC
                                    )
                                best_conf = fc
                                best_emb  = face_bgr
                if best_conf >= 0.85:
                    break
            except Exception:
                continue

        if best_emb is not None and best_conf >= 0.50:
            result = DeepFace.represent(
                best_emb,
                model_name="Facenet512",
                enforce_detection=False,
                detector_backend="opencv"
            )
            if result:
                ref_face_embedding = np.array(result[0]["embedding"])
                ref_face_conf      = best_conf
                face_mode_available = True
                print(f"  ✅ Face embedding ready (conf={best_conf:.2f})")
    except Exception as e:
        print(f"  ⚠️  Face extraction failed: {e}")

    if not face_mode_available:
        print(f"  ⚠️  Face mode unavailable — appearance-only mode")

    # ── 1B: Appearance features ───────────────────────────────
    ref_appearance = extract_person_appearance(ref_image)
    appearance_mode_available = ref_appearance is not None

    if appearance_mode_available:
        ref_upper = ref_appearance.get("upper_color", "unknown")
        ref_lower = ref_appearance.get("lower_color", "unknown")
        print(f"  ✅ Appearance features ready")
        print(f"     Upper={ref_upper}, Lower={ref_lower}")
    else:
        print(f"  ⚠️  Appearance mode unavailable")

    if not face_mode_available and not appearance_mode_available:
        print(f"  ❌ Both modes failed — cannot search")
        results_list.append({"error": "Reference image process nahi hua"})
        return

    # ══════════════════════════════════════════════════════════
    # Thresholds (strict "exact" matching to reduce false positives)
    # ══════════════════════════════════════════════════════════
    # Face thresholds (Facenet512 cosine similarity)
    FACE_HIGH_THRESHOLD  = 0.80   # face alone → save (strict)
    FACE_LOW_THRESHOLD   = 0.70   # face medium → needs appearance support

    # Appearance thresholds (clothing/color/HOG based)
    APPEAR_HIGH_THRESHOLD = 0.82  # appearance alone (fallback, strict)
    APPEAR_LOW_THRESHOLD  = 0.68  # appearance medium → needs face support

    # Combined threshold when both agree but not high
    COMBINED_THRESHOLD = 0.68

    # When face mode is available but the video frame doesn't yield a face (common in CCTV),
    # allow a VERY strict appearance-only fallback instead of rejecting everything.
    # This avoids 0 results when faces are not detectable.
    # If your reference image has no clear face (common), this prevents empty results.
    # Still strict + requires multi-frame confirmation below.
    APPEAR_ONLY_WHEN_NO_FACE_THRESHOLD = 0.84

    # Multi-frame confirmation gate:
    # - if face is available: require 3 hits in same cell
    # - else: 2 hits is enough
    REQUIRED_CONFIRM_HITS = 3 if face_mode_available else 2
    REQUIRED_CONFIRM_HITS_APPEAR_ONLY = 3
    confirm_hits = {}

    print(f"  Thresholds: face_high={FACE_HIGH_THRESHOLD}, "
          f"appear_high={APPEAR_HIGH_THRESHOLD}")
    print(f"{'='*60}\n")

    # ══════════════════════════════════════════════════════════
    # STEP 2: Video frame loop
    # ══════════════════════════════════════════════════════════
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        world_model.set_classes(["person"])
        # Lower conf to improve recall (CCTV person boxes often low confidence)
        res_p = world_model(frame, conf=0.25, imgsz=640)
        if res_p[0].boxes is None:
            continue

        p_boxes = res_p[0].boxes.xyxy.cpu().numpy()

        for pbox in p_boxes:
            px1, py1, px2, py2 = map(int, pbox)

            pw = px2 - px1
            ph = py2 - py1
            if pw * ph < AREA_THRESHOLD or pw < 20 or ph < 40:
                continue

            # Reject weird/oversized/non-person-like boxes (helps exactness a lot)
            if not is_valid_person_box([px1, py1, px2, py2], frame.shape, min_area=max(900, AREA_THRESHOLD)):
                continue

            person_crop = frame[py1:py2, px1:px2]
            if person_crop.size == 0 or is_blurry(person_crop):
                continue

            # ══════════════════════════════════════════════════
            # STEP 3A: Face score
            # ══════════════════════════════════════════════════
            face_score    = 0.0
            face_detected = False

            if face_mode_available and ref_face_embedding is not None:
                try:
                    # Face region — upper 38% of person crop
                    face_h = int(ph * 0.38)
                    if face_h >= 15:
                        face_region = person_crop[:face_h, :]

                        # Upscale small face regions
                        fr_h, fr_w = face_region.shape[:2]
                        if min(fr_h, fr_w) < 80:
                            sc = 80 / min(fr_h, fr_w)
                            face_region = cv2.resize(
                                face_region,
                                (int(fr_w*sc), int(fr_h*sc)),
                                interpolation=cv2.INTER_CUBIC
                            )

                        # CLAHE enhance
                        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
                        clahe = cv2.createCLAHE(clipLimit=2.5,
                                                tileGridSize=(8,8))
                        lab[:,:,0] = clahe.apply(lab[:,:,0])
                        face_region = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                        # Try multiple backends for face detect
                        best_fc   = 0.0
                        best_crop = None
                        for backend in ["opencv", "ssd", "mtcnn"]:
                            try:
                                faces = DeepFace.extract_faces(
                                    face_region,
                                    enforce_detection=False,
                                    detector_backend=backend,
                                    align=True
                                )
                                for fo in faces:
                                    fc = fo.get("confidence", 0.0)
                                    # stricter face detection confidence to reduce wrong face crops
                                    if fc > best_fc and fc >= 0.55:
                                        fa = fo.get("face")
                                        if fa is not None:
                                            best_fc   = fc
                                            best_crop = cv2.cvtColor(
                                                (fa*255).astype(np.uint8),
                                                cv2.COLOR_RGB2BGR
                                            )
                                if best_fc >= 0.75:
                                    break
                            except Exception:
                                continue

                        if best_crop is not None and best_fc >= 0.55:
                            face_detected = True
                            emb_res = DeepFace.represent(
                                best_crop,
                                model_name="Facenet512",
                                enforce_detection=False,
                                detector_backend="opencv"
                            )
                            if emb_res:
                                crop_emb   = np.array(emb_res[0]["embedding"])
                                face_score = float(cosine_similarity(
                                    ref_face_embedding, crop_emb
                                ))
                except Exception:
                    pass

            # ══════════════════════════════════════════════════
            # STEP 3B: Appearance score
            # ══════════════════════════════════════════════════
            appear_score = 0.0
            q_upper      = "unknown"
            q_lower      = "unknown"

            if appearance_mode_available:
                query_app = extract_person_appearance(person_crop)
                if query_app is not None:
                    q_upper = query_app.get("upper_color", "unknown")
                    q_lower = query_app.get("lower_color", "unknown")

                    # Hard gate — upper color mismatch → 0
                    if (ref_upper != "unknown" and
                            q_upper != "unknown" and
                            ref_upper != q_upper):
                        appear_score = 0.0
                    else:
                        appear_score = compare_appearances(
                            ref_appearance, query_app
                        )

            # ══════════════════════════════════════════════════
            # STEP 4: Decision logic
            # ══════════════════════════════════════════════════
            final_score = 0.0
            match_reason = ""
            should_save  = False

            print(f"  Frame {frame_id}: "
                  f"face={face_score:.3f} "
                  f"(det={'✓' if face_detected else '✗'}), "
                  f"appear={appear_score:.3f} "
                  f"[U:{q_upper} L:{q_lower}]")

            # Keep best candidates even if not passing strict thresholds (prevents empty output).
            # Candidate score prioritizes face, else appearance.
            cand_score = face_score if face_detected else appear_score
            if cand_score > 0.0:
                top_candidates.append({
                    "score": float(cand_score),
                    "frame_id": frame_id,
                    "bbox": [px1, py1, px2, py2],
                    "face_score": float(face_score),
                    "appear_score": float(appear_score),
                    "upper_color": q_upper,
                    "lower_color": q_lower,
                    "face_detected": bool(face_detected),
                })

            # If face mode is available but no face is detected in this frame,
            # we still allow a very strict appearance-only fallback (see below).

            # Case 1: Face HIGH alone
            if face_score >= FACE_HIGH_THRESHOLD:
                final_score  = face_score
                match_reason = "face"
                should_save  = True

            # Case 2: Appearance HIGH alone (only when face mode unavailable)
            elif (not face_mode_available) and appear_score >= APPEAR_HIGH_THRESHOLD:
                final_score  = appear_score
                match_reason = "appearance"
                should_save  = True

            # Case 3: Face MEDIUM + Appearance MEDIUM → combined
            elif (face_score >= FACE_LOW_THRESHOLD and
                  appear_score >= COMBINED_THRESHOLD):
                # Weighted average: face 60%, appearance 40%
                final_score  = face_score * 0.6 + appear_score * 0.4
                match_reason = "face+appearance"
                should_save  = True

            # Case 4: Face MEDIUM + Appearance match (upper color same)
            elif (face_score >= FACE_LOW_THRESHOLD and
                  q_upper == ref_upper and
                  ref_upper != "unknown" and
                  appear_score >= APPEAR_LOW_THRESHOLD):
                final_score  = face_score * 0.7 + appear_score * 0.3
                match_reason = "face+color"
                should_save  = True

            # Case 5: Appearance MEDIUM + Face slightly positive
            elif (appear_score >= APPEAR_HIGH_THRESHOLD - 0.05 and
                  face_score >= 0.45 and face_detected):
                final_score  = appear_score * 0.5 + face_score * 0.5
                match_reason = "appearance+weak_face"
                should_save  = True

            # Case 6: Face mode available BUT no face detected → ultra-strict appearance-only fallback
            elif (face_mode_available and ref_face_embedding is not None and not face_detected and
                  appear_score >= APPEAR_ONLY_WHEN_NO_FACE_THRESHOLD and
                  ref_upper != "unknown" and q_upper == ref_upper):
                final_score  = appear_score
                match_reason = "appearance_no_face"
                should_save  = True

            if not should_save:
                reject_count += 1
                print(f"    ⛔ Rejected")
                continue

            # ── Multi-frame confirmation gate ───────────────────────────
            # Only save after we see the same spatial cell hit multiple times.
            ckey = (px1 // 80, py1 // 80, match_reason)
            confirm_hits[ckey] = confirm_hits.get(ckey, 0) + 1
            required_hits = (
                REQUIRED_CONFIRM_HITS_APPEAR_ONLY
                if match_reason == "appearance_no_face"
                else REQUIRED_CONFIRM_HITS
            )
            if confirm_hits[ckey] < required_hits:
                print(f"    ⏳ Pending confirm {confirm_hits[ckey]}/{required_hits}")
                continue

            # ══════════════════════════════════════════════════
            # STEP 5: Dedup + Save
            # ══════════════════════════════════════════════════
            key = f"ref_{px1//60}_{py1//60}"
            if tracker.should_skip(key, frame_id):
                continue
            tracker.update(key, frame_id)

            match_count += 1
            conf_pct = int(final_score * 100)

            # Color-code box by match type
            if "face" in match_reason and "appearance" in match_reason:
                box_color = (0, 255, 0)    # Green — both matched
            elif match_reason == "face":
                box_color = (0, 200, 255)  # Yellow — face only
            else:
                box_color = (255, 165, 0)  # Orange — appearance only

            label = f"MATCH {conf_pct}% [{match_reason}]"

            ann = frame.copy()

            # Detection box
            cv2.rectangle(ann, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(ann, label,
                        (px1, max(py1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        box_color, 2)

            # Confidence bar
            if py1 > 20:
                bar_fill = int((px2 - px1) * final_score)
                cv2.rectangle(ann, (px1, py1-16),
                              (px2, py1-4), (40, 40, 40), -1)
                cv2.rectangle(ann, (px1, py1-16),
                              (px1 + bar_fill, py1-4),
                              box_color, -1)

            # Score breakdown overlay
            cv2.putText(ann,
                        f"F:{face_score:.2f} A:{appear_score:.2f}",
                        (px1, py2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (200, 200, 200), 1)

            # Reference thumbnail — top right
            try:
                th, tw = 90, 60
                thumb = cv2.resize(ref_image, (tw, th))
                fh2, fw2 = ann.shape[:2]
                ann[10:10+th, fw2-tw-10:fw2-10] = thumb
                cv2.rectangle(ann,
                              (fw2-tw-10, 10),
                              (fw2-10, 10+th),
                              box_color, 2)
                cv2.putText(ann, "REF",
                            (fw2-tw-8, 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, box_color, 1)
            except Exception:
                pass

            img_path = os.path.join(
                SAVE_DIR,
                f"ref_match_{get_session_id()}_{match_count}.jpg"
            )
            cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 90])

            results_list.append({
                "object":        "Reference Person Found",
                "match_type":    match_reason,
                "confidence":    final_score,
                "face_score":    face_score,
                "appear_score":  appear_score,
                "upper_color":   q_upper,
                "lower_color":   q_lower,
                "person_bbox":   [px1, py1, px2, py2],
                "image_path":    img_path,
                "timestamp":     frame_id,
                "source":        "dual_mode_search"
            })

            print(f"  ✅ MATCH #{match_count} | "
                  f"frame={frame_id} | "
                  f"score={final_score:.3f} ({conf_pct}%) | "
                  f"reason={match_reason}")

    print(f"\n  📊 Summary:")
    print(f"     Total matches : {match_count}")
    print(f"     Rejected      : {reject_count}")
    print(f"{'='*60}\n")

    # If we got 0 strict matches, return top candidates as "candidates" (still useful for user).
    if match_count == 0 and len(top_candidates) > 0:
        top_candidates.sort(key=lambda x: x["score"], reverse=True)
        # Deduplicate by coarse spatial cell and take top 3
        picked = []
        used_cells = set()
        for c in top_candidates:
            x1, y1, x2, y2 = c["bbox"]
            cell = (x1 // 80, y1 // 80)
            if cell in used_cells:
                continue
            used_cells.add(cell)
            picked.append(c)
            if len(picked) >= 3:
                break

        # Rewind video to extract frames for these candidates
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            target_ids = {c["frame_id"] for c in picked}
            fid = 0
            while cap.isOpened() and target_ids:
                ret, frame = cap.read()
                if not ret:
                    break
                fid += 1
                if fid not in target_ids:
                    continue
                for c in picked:
                    if c["frame_id"] != fid:
                        continue
                    x1, y1, x2, y2 = c["bbox"]
                    ann = frame.copy()
                    cv2.rectangle(ann, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(
                        ann,
                        f"CANDIDATE {int(c['score']*100)}% (F:{c['face_score']:.2f} A:{c['appear_score']:.2f})",
                        (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )
                    img_path = os.path.join(
                        SAVE_DIR,
                        f"ref_candidate_{get_session_id()}_{fid}.jpg"
                    )
                    cv2.imwrite(img_path, ann, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    results_list.append({
                        "object": "Reference Person Candidate",
                        "match_type": "candidate",
                        "confidence": float(c["score"]),
                        "face_score": float(c["face_score"]),
                        "appear_score": float(c["appear_score"]),
                        "upper_color": c["upper_color"],
                        "lower_color": c["lower_color"],
                        "person_bbox": [x1, y1, x2, y2],
                        "image_path": img_path,
                        "timestamp": fid,
                        "source": "candidate_fallback"
                    })
                    target_ids.discard(fid)
        except Exception as e:
            print(f"  ⚠️  Candidate fallback failed: {e}")


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
# IMAGE PROCESSING FUNCTION
# ================================

def process_single_image(frame, prompt, parsed, mode):
    """Process single image with all detection modes"""
    results = []
   
    if mode == "auto":
        # AUTO: run a small, practical default set for single images.
        # Keep it conservative to avoid unnecessary detections.
        # Persons
        world_model.set_classes(["person"])
        res_p = world_model(frame, conf=0.40, imgsz=640)
        if res_p[0].boxes is not None:
            for b in res_p[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, b)
                if (x2 - x1) * (y2 - y1) < AREA_THRESHOLD:
                    continue
                results.append({
                    "object": "person",
                    "bbox": [x1, y1, x2, y2],
                    "image_path": save_detection(frame, [x1, y1, x2, y2], "person", 1, "img_auto_person")
                })

        # Vehicles (COCO): car, motorcycle, bus, truck
        yolo_res = car_model(frame, classes=[2, 3, 5, 7], conf=0.40)
        if yolo_res[0].boxes is not None:
            for b in yolo_res[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, b)
                if (x2 - x1) * (y2 - y1) < AREA_THRESHOLD:
                    continue
                results.append({
                    "object": "vehicle",
                    "bbox": [x1, y1, x2, y2],
                    "image_path": save_detection(frame, [x1, y1, x2, y2], "vehicle", 1, "img_auto_vehicle")
                })

        # Helmet (optional, only if persons exist)
        if any(str(r.get("object", "")).lower() == "person" for r in results):
            world_model.set_classes(["helmet"])
            res_h = world_model(frame, conf=0.35, imgsz=640)
            if res_h[0].boxes is not None:
                for b in res_h[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, b)
                    if (x2 - x1) * (y2 - y1) < 250:
                        continue
                    results.append({
                        "object": "helmet",
                        "bbox": [x1, y1, x2, y2],
                        "image_path": save_detection(frame, [x1, y1, x2, y2], "helmet", 1, "img_auto_helmet")
                    })

        # Apply strict post-filter only when user asked a specific prompt.
        # For auto, we intentionally keep the raw conservative results.
        # Dedupe still applies below.
        pass

    if mode == "color_object":
        color_name = normalize_color_name(parsed.get("color") or "")
        for box in detect_vehicles_in_frame(frame, parsed["vehicle_info"], conf_yolo=0.40):
            x1,y1,x2,y2 = box
            if (x2-x1)*(y2-y1) < AREA_THRESHOLD: continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or is_blurry(crop): continue
            if color_name and vehicle_color_match(frame, [x1, y1, x2, y2], color_name):
                label = f"{color_name} {parsed.get('vehicle_word','vehicle')}"
                results.append({
                    "object": label,
                    "color": color_name,
                    "vehicle": parsed.get("vehicle_word", ""),
                    "bbox": [x1,y1,x2,y2],
                    "image_path": save_detection(frame,[x1,y1,x2,y2],label,1,"img_color")
                })

    elif mode == "person_without_helmet":
        # Image mode must set classes explicitly (otherwise YOLOWorld reuses old classes).
        world_model.set_classes(["person"])
        p_boxes = world_model(frame, conf=0.35, imgsz=640)[0].boxes
        p_boxes = p_boxes.xyxy.cpu().numpy() if p_boxes is not None else []

        world_model.set_classes(["helmet"])
        h_boxes = world_model(frame, conf=0.35, imgsz=640)[0].boxes
        h_boxes = h_boxes.xyxy.cpu().numpy() if h_boxes is not None else []

        # Two-wheeler: prefer YOLO class-3, fallback to world scooter/motorcycle when class-3 misses.
        v_boxes = []
        yolo_res = car_model(frame, classes=[3], conf=0.40)
        if yolo_res[0].boxes is not None:
            for b in yolo_res[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, b)
                w = x2 - x1
                h = y2 - y1
                area = w * h
                aspect_wh = w / (h + 1e-5)
                if 0.85 <= aspect_wh <= 4.20 and 900 < area < 45000:
                    v_boxes.append([x1, y1, x2, y2])
        if not v_boxes:
            for label in ["scooter", "motorcycle", "motorbike"]:
                world_model.set_classes([label])
                res_v = world_model(frame, conf=0.42, imgsz=640)
                if res_v[0].boxes is None:
                    continue
                for b in res_v[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, b)
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    aspect_wh = w / (h + 1e-5)
                    if 0.85 <= aspect_wh <= 4.20 and 900 < area < 45000:
                        candidate = [x1, y1, x2, y2]
                        if not any(iou(candidate, ub) > 0.55 for ub in v_boxes):
                            v_boxes.append(candidate)
       
        for pbox in p_boxes:
            px1,py1,px2,py2 = map(int, pbox)
            if (px2-px1)*(py2-py1) < AREA_THRESHOLD: continue
           
            has_helmet = False
            for h in h_boxes:
                if helmet_belongs_to_person_strict(
                    [px1,py1,px2,py2],
                    [int(h[0]),int(h[1]),int(h[2]),int(h[3])]
                ):
                    has_helmet = True
                    break
            if has_helmet: continue
           
            person_box = [px1, py1, px2, py2]
            matched_tw = None
            for vbox in v_boxes:
                if is_person_on_vehicle(person_box, vbox):
                    matched_tw = vbox
                    break
            if matched_tw is None:
                continue
           
            results.append({
                "object": "NO HELMET! (Two-Wheeler)",
                "person_bbox": [px1,py1,px2,py2],
                "vehicle_bbox": matched_tw,
                "image_path": save_detection(frame,[px1,py1,px2,py2],"NO HELMET!",1,"img_no_helmet",(0,0,255))
            })

    elif mode == "triple_riding":
        vehicle_type = parsed.get("vehicle_type", "two_wheeler")
        if vehicle_type == "bike":
            v_info = {"strategy": "world", "world_label": "bike"}
        elif vehicle_type == "scooter":
            v_info = {"strategy": "world", "world_label": "scooter"}
        else:
            v_info = {"strategy": "yolo_class", "class_ids": [3]}
       
        p_boxes = world_model(frame, conf=0.45, imgsz=640)[0].boxes
        if p_boxes is not None:
            p_boxes = p_boxes.xyxy.cpu().numpy()
        else:
            p_boxes = []
       
        v_boxes = detect_vehicles_in_frame(frame, v_info, conf_yolo=0.30, conf_world=0.35)
       
        triple_count = 0
        for vbox in v_boxes:
            vx1,vy1,vx2,vy2 = vbox
            riders = [p for p in p_boxes
                     if max(0, min(p[2],vx2)-max(p[0],vx1))/(vx2-vx1) > 0.25]
           
            if len(riders) >= 3:
                ann = frame.copy()
                draw_box(ann, vx1,vy1,vx2,vy2, f"{len(riders)} riders", (255,0,255))
                for r in riders:
                    draw_box(ann, int(r[0]),int(r[1]),int(r[2]),int(r[3]),"rider",(0,255,255))
                triple_count += 1
                img_path = os.path.join(SAVE_DIR, f"img_triple_{get_session_id()}_{triple_count}.jpg")
                cv2.imwrite(img_path, ann)
                results.append({
                    "object": f"TRIPLE RIDING! ({len(riders)} persons)",
                    "person_count": len(riders),
                    "image_path": img_path
                })

    elif mode == "person_helmet_any":
        world_model.set_classes(["person"])
        p_boxes = world_model(frame, conf=0.40, imgsz=640)[0].boxes
        if p_boxes is None: p_boxes = []
        else: p_boxes = p_boxes.xyxy.cpu().numpy()
       
        world_model.set_classes(["helmet"])
        h_boxes = world_model(frame, conf=0.35, imgsz=640)[0].boxes
        if h_boxes is not None:
            h_boxes = h_boxes.xyxy.cpu().numpy()
        else:
            h_boxes = []
       
        for pbox in p_boxes:
            for hbox in h_boxes:
                if helmet_belongs_to_person_strict([int(p) for p in pbox],[int(h) for h in hbox]):
                    results.append({
                        "object": "person with helmet",
                        "person_bbox": [int(p) for p in pbox],
                        "image_path": save_detection(frame,[int(p) for p in pbox],"person with helmet",1,"img_helmet")
                    })

    elif mode == "person_bag_any":
        bag_type = parsed.get("bag_type", "bag")
        world_model.set_classes(["person"])
        p_boxes = world_model(frame, conf=0.40, imgsz=640)[0].boxes
        if p_boxes is None: p_boxes = []
        else: p_boxes = p_boxes.xyxy.cpu().numpy()

        # Use the same bag detector used in video mode (COCO + YOLOWorld fallback).
        b_boxes = detect_bags_in_frame(frame, bag_type, conf=0.22)
       
        for pbox in p_boxes:
            for bbox in b_boxes:
                if bag_belongs_to_person_strict([int(p) for p in pbox],[int(b) for b in bbox]):
                    results.append({
                        "object": f"person with {bag_type}",
                        "person_bbox": [int(p) for p in pbox],
                        "image_path": save_detection(frame,[int(p) for p in pbox],f"person with {bag_type}",1,"img_bag")
                    })

    elif mode == "person_shoes_any":
        shoe_type = normalize_shoe_type(parsed.get("shoe_type", "shoes"))
        world_model.set_classes(["person"])
        p_boxes = world_model(frame, conf=0.40, imgsz=640)[0].boxes
        if p_boxes is None: p_boxes = []
        else: p_boxes = p_boxes.xyxy.cpu().numpy()

        footwear_detections = detect_footwear_in_frame(frame, shoe_type=shoe_type, conf=0.18)
       
        for pbox in p_boxes:
            pb = [int(p) for p in pbox]
            for (sbox, shoe_label) in footwear_detections:
                sb = [int(s) for s in sbox]
                if not shoes_belongs_to_person_strict(pb, sb):
                    continue
                if not is_plausible_shoe_box_for_person(pb, sb):
                    continue
                    results.append({
                        "object": "person with shoes",
                        "person_bbox": pb,
                        "shoe_type": shoe_type,
                        "image_path": save_detection(frame,[int(p) for p in pbox],"person with shoes",1,"img_shoes")
                    })

    elif mode == "person_clothing_any":
        clothing = parsed.get("clothing_type", "shirt")
        clothing = normalize_clothing_type(clothing)
        world_model.set_classes(["person"])
        p_boxes = world_model(frame, conf=0.40, imgsz=640)[0].boxes
        if p_boxes is None: p_boxes = []
        else: p_boxes = p_boxes.xyxy.cpu().numpy()

        # Detect clothing labels (use expanded label list).
        yolo_labels = CLOTHING_YOLO_LABELS.get(clothing.lower(), [clothing])
        c_boxes = []
        for yl in yolo_labels:
            world_model.set_classes([yl])
            cres = world_model(frame, conf=0.18, imgsz=640)
            if cres[0].boxes is None:
                continue
            for b in cres[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, b)
                if (x2 - x1) * (y2 - y1) < 250:
                    continue
                candidate = [x1, y1, x2, y2]
                if not any(iou(candidate, ub) > 0.55 for ub in c_boxes):
                    c_boxes.append(candidate)
       
        for pbox in p_boxes:
            for cbox in c_boxes:
                if clothing_belongs_to_person_strict([int(p) for p in pbox],[int(c) for c in cbox], clothing_type=clothing):
                    results.append({
                        "object": f"person wearing {clothing}",
                        "person_bbox": [int(p) for p in pbox],
                        "clothing_type": clothing,
                        "image_path": save_detection(frame,[int(p) for p in pbox],f"person wearing {clothing}",1,"img_cloth")
                    })

    else:
        classes = [prompt.strip().lower()]
        world_model.set_classes(classes)
        res = world_model(frame, conf=0.40, imgsz=640)
        if res[0].boxes is not None:
            for box in res[0].boxes.xyxy.cpu().numpy():
                x1,y1,x2,y2 = map(int, box)
                if (x2-x1)*(y2-y1) < AREA_THRESHOLD: continue
                label = prompt
                results.append({
                    "object": label,
                    "bbox": [x1,y1,x2,y2],
                    "image_path": save_detection(frame,[x1,y1,x2,y2],label,1,"img_generic")
                })

    # Apply the same strict post-filter used in video pipeline
    # only when a specific intent exists.
    if mode != "auto":
        results = enforce_prompt_exactness(parsed, results)

    # Dedupe: keep unique detections by IoU for single-image results.
    def _result_box(r):
        return r.get("bbox") or r.get("person_bbox") or r.get("vehicle_bbox") or []

    unique = []
    for r in results:
        b = _result_box(r)
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            unique.append(r)
            continue
        if any(iou(b, _result_box(u)) > 0.60 for u in unique if isinstance(_result_box(u), (list, tuple)) and len(_result_box(u)) == 4):
            continue
        unique.append(r)

    print(f"📊 Image Results: {len(unique)}")
    return unique


# ================================
# MAIN API - UPDATED
# ================================

@app.post("/process")
async def process_video(req: Request):
    global SESSION_ID
    SESSION_ID = str(int(__import__('time').time() * 1000))
   
    data = await req.json()

    video_url = data.get("fileUrl")
    prompt    = data.get("prompt", "person")
    image_url = data.get("imageUrl")

    print(f"\n{'='*60}")
    print(f"🔍 REQUEST RECEIVED")
    print(f"   Prompt: {prompt}")
    print(f"   Video URL: {video_url}")
    print(f"   Image URL: {image_url}")
    print(f"{'='*60}")

    parsed = parse_prompt(prompt)
    mode   = parsed["mode"]
    print(f"\n✅ Parsed Result:")
    print(f"   Mode: {mode}")
    print(f"   Parsed Data: {parsed}")
    print(f"{'='*60}")

    results_list = []

    # ============================
    # REFERENCE IMAGE FEATURES
    # ============================
    ref_feat = None
    ref_image = None
    ref_image_person = None
   
    if image_url and image_url.strip() and image_url.strip().lower() not in ["null", "undefined", "none", "image.png", "image.jpg", "image.jpeg"]:
        print(f"🖼️ Downloading REFERENCE IMAGE...")
        try:
            r = requests.get(image_url, timeout=30)
            if r.status_code == 200:
                nparr = np.frombuffer(r.content, np.uint8)
                ref_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
               
                if ref_image is not None:
                    print(f"✅ Reference image downloaded ({ref_image.shape[1]}x{ref_image.shape[0]})")

                    # 🔥 IMPORTANT: crop the reference PERSON first (reduces background noise)
                    # Your uploaded "reference image" is usually a full frame from CCTV/video.
                    # Matching works much better when we extract features from the person crop.
                    try:
                        world_model.set_classes(["person"])
                        rref = world_model(ref_image, conf=0.35, imgsz=640)
                        best_box = None
                        best_area = 0
                        if rref and rref[0].boxes is not None:
                            for b in rref[0].boxes.xyxy.cpu().numpy():
                                x1, y1, x2, y2 = map(int, b)
                                # clamp
                                x1 = max(0, x1); y1 = max(0, y1)
                                x2 = min(ref_image.shape[1] - 1, x2)
                                y2 = min(ref_image.shape[0] - 1, y2)
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                area = (x2 - x1) * (y2 - y1)
                                if area < max(900, AREA_THRESHOLD):
                                    continue
                                # prefer a valid person-like box
                                if not is_valid_person_box([x1, y1, x2, y2], ref_image.shape, min_area=max(900, AREA_THRESHOLD)):
                                    continue
                                if area > best_area:
                                    best_area = area
                                    best_box = (x1, y1, x2, y2)

                        if best_box is not None:
                            x1, y1, x2, y2 = best_box
                            # a little padding helps include clothing context
                            pad_x = int((x2 - x1) * 0.08)
                            pad_y = int((y2 - y1) * 0.08)
                            x1p = max(0, x1 - pad_x)
                            y1p = max(0, y1 - pad_y)
                            x2p = min(ref_image.shape[1] - 1, x2 + pad_x)
                            y2p = min(ref_image.shape[0] - 1, y2 + pad_y)
                            ref_image_person = ref_image[y1p:y2p, x1p:x2p]
                            print(f"✅ Reference person crop used ({ref_image_person.shape[1]}x{ref_image_person.shape[0]})")
                        else:
                            ref_image_person = ref_image
                            print("⚠️  No person crop found in reference image; using full image")
                    except Exception as e:
                        ref_image_person = ref_image
                        print(f"⚠️  Reference person crop failed: {e}")

                    # Extract features from reference image (cropped)
                    ref_feat = extract_features(ref_image_person)
                    print(f"✅ Reference features extracted (cropped)")
                   
                    # Check if this is ONLY image search (no video)
                    if not video_url or not video_url.strip() or video_url.strip().lower() in ["null", "undefined", "none", "image.png"]:
                        print(f"🖼️ Processing ONLY reference image...")
                        results_list = process_single_image(ref_image_person, prompt, parsed, mode)
                        return {
                            "results": results_list,
                            "mode": mode,
                            "parsed_prompt": parsed,
                            "total_found": len(results_list),
                            "source": "reference_image_only"
                        }
        except Exception as e:
            print(f"Reference image download failed: {e}")

    # ============================
    # VIDEO PROCESSING
    # ============================
    if not video_url or not video_url.strip():
        return {"error": "Video URL is required", "details": "fileUrl is missing or empty"}
   
    invalid_urls = ["image.png", "image.jpg", "image.jpeg", "null", "undefined", "none"]
    if video_url.strip().lower() in invalid_urls:
        return {"error": "Invalid video file", "details": f"Please provide a valid video file URL"}
   
    if not video_url.startswith(("http://", "https://")):
        return {"error": "Invalid video URL", "details": f"URL must start with http:// or https://"}

    try:
        r = requests.get(video_url, stream=True, timeout=30)
        if r.status_code != 200:
            return {"error": f"Failed to download video", "details": f"HTTP {r.status_code}"}
       
        with open("temp_video.mp4", "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        print(f"✅ Video downloaded successfully")
    except Exception as e:
        return {"error": f"Failed to download video", "details": str(e)}

    cap = cv2.VideoCapture("temp_video.mp4")
    if not cap.isOpened():
        return {"error": "video open failed - file may be corrupted"}

    print(f"  Video ready for processing")

    if ref_feat is not None and ref_image is not None:

        print(f"🔍 Running REFERENCE IMAGE SEARCH (highest priority)")
        run_reference_image_search_mode(cap, ref_feat, (ref_image_person if ref_image_person is not None else ref_image), results_list)
   
    elif mode == "person_clothing_color":
        run_person_clothing_color_mode(
            cap,
            parsed["clothing_type"],
            parsed["color"],
            results_list
        )
   
    elif mode == "person_clothing_any":
        run_person_clothing_any_mode(
            cap,
            parsed["clothing_type"],
            results_list
        )

    elif mode == "person_color_only":
        run_person_color_only_mode(cap, parsed["color"], parsed.get("gender"), results_list)

    elif mode == "person_shoes_any":
        shoe_type = parsed.get("shoe_type", "shoes")
        run_person_shoes_any_mode(cap, shoe_type, results_list)  # ✅ 3 args

    elif mode == "person_shoes_color":
        shoe_type = parsed.get("shoe_type", "shoes")
        run_person_shoes_color_mode(cap, parsed["color"], shoe_type, results_list)

    elif mode == "person_helmet_color":
        run_person_helmet_color_mode(
        cap, parsed["color"], results_list)

    elif mode == "person_helmet_any":
        run_person_helmet_any_mode(cap, prompt,results_list)

    elif mode == "person_without_helmet":
        run_person_without_helmet_mode(cap, results_list)

    elif mode == "triple_riding":
        run_triple_riding_mode(cap, parsed.get("vehicle_type", "two_wheeler"), results_list)

    elif mode == "plate":
        run_plate_mode(cap, parsed["plate_info"], results_list)
   


 


    elif mode == "text_search":
        run_text_search_mode(
        cap,
        parsed["target_text"],
        results_list
    )
       


    elif mode == "person_bag_any":
        run_person_bag_any_mode(
            cap,
            parsed["bag_type"],
            results_list
        )

 
   

    elif mode == "person_bag_color":
        run_person_bag_color_mode(
            cap,
            parsed["color"],
            parsed["bag_type"],
            results_list
    )


    elif mode == "gender_color":
        run_gender_color_mode(
        cap,
        parsed["gender"],
        parsed["color"],
        results_list
    )
   
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
        # Only run yoloworld if NOT a color+vehicle query
        detected_color = extract_color_from_prompt(prompt.lower())
        veh_word, veh_info = find_vehicle_in_prompt(prompt.lower())
       
        if detected_color and veh_word:
            print(f"  ERROR: Should have been color_object mode but got '{mode}'!")
            print(f"  Forcing color_object mode for: {detected_color} {veh_word}")
            # Force color_object even if mode is wrong
            run_color_object_mode(cap, detected_color, veh_word, veh_info, results_list)
        else:
            run_yoloworld_mode(cap, prompt, ref_feat, results_list)

    cap.release()

    # Final strict guard: drop any result that does not match parsed prompt intent.
    results_list = enforce_prompt_exactness(parsed, results_list)

    print(f"\n📊 Results found: {len(results_list)}")
    print(f"{'='*60}\n")

    return {
        "results":       results_list,
        "mode":          mode,
        "parsed_prompt": parsed,
        "total_found":   len(results_list)
    }

