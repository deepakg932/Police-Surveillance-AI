



# from fastapi import FastAPI, Request
# from ultralytics import YOLO
# import cv2

# app = FastAPI()

# # 👤 person model
# person_model = YOLO("yolov8s.pt")

# # 🪖 helmet model (download karke rakh)
# helmet_model = YOLO("helmet.pt")

# print(helmet_model.names)


# @app.post("/process")
# async def process_video(req: Request):
#     data = await req.json()
#     filePath = data.get("filePath")

#     cap = cv2.VideoCapture(filePath)

#     if not cap.isOpened():
#         return {"results": []}

#     results_list = []
#     frame_id = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_id += 1

#         # 👤 detect persons
#         person_results = person_model.track(frame, persist=True)

#         for r in person_results:
#             for box in r.boxes:
#                 cls_id = int(box.cls[0])
#                 name = person_model.names[cls_id]

#                 if name != "person":
#                     continue

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = float(box.conf[0])

#                 height = y2 - y1

#                 # 🪖 head crop
#                 head_y1 = y1
#                 head_y2 = y1 + int(height * 0.35)

#                 head_crop = frame[head_y1:head_y2, x1:x2]

#                 helmet = False

#                 if head_crop is not None and head_crop.size != 0:
#                     helmet_results = helmet_model(head_crop)

#                     for hr in helmet_results:
#                         for hbox in hr.boxes:
#                             h_cls = int(hbox.cls[0])
#                             h_name = helmet_model.names[h_cls]

#                             if "helmet" in h_name.lower():
#                                 helmet = True
#                                 break

#                 tracking_id = f"{x1}_{y1}_{x2}_{y2}"

#                 results_list.append({
#                     "object": "person",
#                     "color": "",
#                     "helmet": helmet,
#                     "vehicle": "",
#                     "timestamp": str(frame_id),
#                     "confidence": conf,
#                     "trackingId": tracking_id
#                 })

#     cap.release()

#     return {"results": results_list} 


from fastapi import FastAPI, Request
from ultralytics import YOLOWorld
import cv2
import os

app = FastAPI()
model = YOLOWorld("yolov8s-world.pt")

# 📁 Folder jahan screenshots save honge
SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/process")
async def process_video(req: Request):
    data = await req.json()
    filePath = data.get("filePath")
    user_prompt = data.get("prompt", "person, car, helmet") 
    
    custom_classes = [c.strip() for c in user_prompt.split(",")]
    model.set_classes(custom_classes)

    cap = cv2.VideoCapture(filePath)
    results_list = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        
        if frame_id % 15 != 0: continue # Speed ke liye (har 15th frame)

        results = model.predict(frame, conf=0.7)
        
        detected_in_this_frame = False
        
        for r in results:
            if not r.boxes: continue
            
            # Frame par boxes draw karne ke liye hum original frame ki copy lenge
            annotated_frame = frame.copy()

            for box in r.boxes:
                detected_in_this_frame = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = custom_classes[cls_id]

                # 🎨 Visual Proof: Frame par rectangle draw karo
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                results_list.append({
                    "object": label,
                    "confidence": conf,
                    "trackingId": f"{x1}_{y1}", 
                    "timestamp": str(frame_id),
                    "image_path": f"{SAVE_DIR}/frame_{frame_id}.jpg"
                })

            # Agar kuch detect hua, toh image save karo
            if detected_in_this_frame:
                img_name = f"frame_{frame_id}.jpg"
                cv2.imwrite(os.path.join(SAVE_DIR, img_name), annotated_frame)
                print(f"✅ Detected {label} at frame {frame_id}")

    cap.release()
    return {"results": results_list}