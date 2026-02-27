"""
Face detection + demographics pipeline.
Detection:    YOLOv8 (arnabdhar/YOLOv8-Face-Detection)
Demographics: AgeGenderViTModel (abhilash88/age-gender-prediction)
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image

from model import predict_age_gender

# ── 1. Load image ────────────────────────────────────────────────────────────
IMAGE_PATH = "/tmp/test_face.jpg"
image = Image.open(IMAGE_PATH).convert("RGB")
W, H = image.size
print(f"Image loaded: {image.size}")

# ── 2. YOLO face detection ────────────────────────────────────────────────────
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
detector = YOLO(model_path)

output = detector(image)
detections = Detections.from_ultralytics(output[0])

print(f"\nYOLO found {len(detections)} face(s)")

if len(detections) == 0:
    print("No faces detected. Exiting.")
    sys.exit(1)

# ── 3. Select largest face by bounding box area ───────────────────────────────
# detections.xyxy is [x1, y1, x2, y2] in pixel coords
boxes = detections.xyxy  # shape (N, 4)
areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
largest_idx = areas.argmax()
x1, y1, x2, y2 = boxes[largest_idx].tolist()

# add 20% padding
pad_x = (x2 - x1) * 0.2
pad_y = (y2 - y1) * 0.2
x1 = max(0, int(x1 - pad_x))
y1 = max(0, int(y1 - pad_y))
x2 = min(W, int(x2 + pad_x))
y2 = min(H, int(y2 + pad_y))

face_crop = image.crop((x1, y1, x2, y2))
print(f"  Largest face region: ({x1},{y1}) → ({x2},{y2})")

# ── 4. Demographics inference ─────────────────────────────────────────────────
print("\nRunning demographics model...")
result = predict_age_gender(face_crop)

print("\nResults:")
print("-" * 40)
print(f"  Age:    {result['age']}")
print(f"  Gender: {result['gender']}  ({result['gender_confidence']:.1%})")
