# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from model import predict_age_gender
from pathlib import Path
import os
import time
import shutil

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO("model.pt")

# inference

directory_path = Path('images')


for filename in os.listdir(directory_path):
	full_path = os.path.join(directory_path, filename)
	if os.path.isfile(full_path):
		output = model(Image.open(full_path))
		results = Detections.from_ultralytics(output[0])

		start = time.time()
		result = predict_age_gender(full_path)
		end = time.time()
		elapsed = end - start
		print(f"Age: {result['age']}, Gender: {result['gender']}, Time Elapsed: {elapsed} seconds")

