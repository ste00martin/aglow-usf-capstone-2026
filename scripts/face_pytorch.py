"""
Face detection + demographics using pure PyTorch models.
Detection: BlazeFace (blazeface.pth)
Demographics: HuggingFace ViT models (age + gender)
"""

import sys
import torch
import numpy as np
from PIL import Image
from transformers import pipeline

# ── 1. Load image ────────────────────────────────────────────────────────────
IMAGE_PATH = "/tmp/test_face.jpg"
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"Image loaded: {image.size}")

# ── 2. BlazeFace detection ────────────────────────────────────────────────────
sys.path.insert(0, ".")  # so we can import blazeface.py from the same folder
from blazeface import BlazeFace

device = torch.device("cpu")
detector = BlazeFace().to(device)
detector.load_weights("blazeface.pth")
# numpy 2.x requires allow_pickle=True for older .npy files
anchors = np.load("anchors.npy", allow_pickle=True)
detector.anchors = torch.tensor(anchors, dtype=torch.float32).to(device)
detector.min_score_thresh = 0.75

# BlazeFace expects a 128x128 RGB image, values in [0, 1]
img_resized = image.resize((128, 128))
# BlazeFace's _preprocess does x/127.5 - 1.0, so pass raw [0,255] values
img_tensor = torch.tensor(np.array(img_resized), dtype=torch.float32)
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, 128, 128]

detections = detector.predict_on_batch(img_tensor)
faces_found = detections[0]  # detections for first (only) image

print(f"\nBlazeFace found {len(faces_found)} face(s)")

# ── 3. Crop faces from original image ────────────────────────────────────────
W, H = image.size
face_crops = []

for det in faces_found:
    # det[:4] = [ymin, xmin, ymax, xmax] in normalized coords
    ymin, xmin, ymax, xmax = det[:4].tolist()
    # add 20% padding around the face
    pad_x = (xmax - xmin) * 0.2
    pad_y = (ymax - ymin) * 0.2
    x1 = max(0, int((xmin - pad_x) * W))
    y1 = max(0, int((ymin - pad_y) * H))
    x2 = min(W, int((xmax + pad_x) * W))
    y2 = min(H, int((ymax + pad_y) * H))
    crop = image.crop((x1, y1, x2, y2))
    face_crops.append(crop)
    print(f"  Cropped face region: ({x1},{y1}) → ({x2},{y2})")

if not face_crops:
    print("No faces detected. Exiting.")
    sys.exit(1)

# ── 4. Demographics inference ─────────────────────────────────────────────────
print("\nLoading demographics models from HuggingFace...")

age_pipe    = pipeline("image-classification", model="nateraw/vit-age-classifier")
gender_pipe = pipeline("image-classification", model="rizvandwiki/gender-classification")

print("\nResults:")
print("-" * 40)

for i, crop in enumerate(face_crops):
    age_out    = age_pipe(crop)
    gender_out = gender_pipe(crop)

    top_age    = age_out[0]["label"]
    top_gender = gender_out[0]["label"]
    age_conf   = age_out[0]["score"]
    gender_conf = gender_out[0]["score"]

    print(f"Face {i + 1}:")
    print(f"  Age:    {top_age}  ({age_conf:.1%})")
    print(f"  Gender: {top_gender}  ({gender_conf:.1%})")
