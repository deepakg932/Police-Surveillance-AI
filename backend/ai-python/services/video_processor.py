"""
Smart frame extraction - video kitna bhi bada/ chhota ho, properly detect.
Speed + accuracy optimized.
"""
import os
import base64
import cv2


def extract_frames(
    video_path: str,
    max_frames: int = 12,
    max_width: int = 720,
    jpeg_quality: int = 85,
) -> list[dict]:
    """
    Video ke poore duration se evenly spread frames extract karo.
    Short/long dono ke liye adaptive, fast (direct frame jump).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration_sec = total_frames / fps if fps > 0 else 0

    if total_frames <= 0:
        cap.release()
        return []

    # Video length ke hisaab se frames - short video = zyada, long = kam but spread
    if duration_sec < 30:
        n_frames = min(max_frames, max(6, total_frames // 5))  # short: dense
    elif duration_sec < 120:
        n_frames = min(max_frames, 10)
    else:
        n_frames = min(8, max_frames)  # long: 8 hi kaafi

    n_frames = min(n_frames, total_frames)

    # Evenly spread frame indices - start, middle, end sab cover
    if n_frames >= total_frames:
        indices = list(range(total_frames))
    else:
        step = (total_frames - 1) / (n_frames - 1) if n_frames > 1 else 0
        indices = [int(i * step) for i in range(n_frames)]

    frames = []
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize for speed - chhota image = fast API + fast encode
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w, new_h = max_width, int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode(".jpg", frame, encode_params)
        b64 = base64.standard_b64encode(buffer).decode("utf-8")
        timestamp_sec = round(frame_idx / fps, 1) if fps > 0 else frame_idx
        frames.append({
            "base64": b64,
            "index": i,
            "frame_number": frame_idx,
            "timestamp_sec": timestamp_sec,
        })

    cap.release()
    return frames
