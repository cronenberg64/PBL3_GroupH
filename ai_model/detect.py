"""
ai_model.detect

Handles cat detection using the YOLOv8 model. Given an image path, this module
detects the cat in the image, crops the bounding box, and resizes it for
further processing.

Dependencies:
    - Ultralytics YOLOv8
    - OpenCV

Functions:
    - preprocess_image(image_path, target_size=(224, 224)): Detects and crops a cat image.
"""


import cv2
from ultralytics import YOLO
import os

# Use pre-trained general object detector
model = YOLO("yolov8n.pt")  # Downloads automatically if missing


def preprocess_image(image_path, target_size=(224, 224)):
    results = model(image_path)

    if not results or len(results[0].boxes) == 0:
        raise ValueError("No cat detected.")

    # Filter for 'cat' class (class 15 in COCO)
    cat_class_id = 15
    boxes = results[0].boxes
    cat_indices = [i for i, c in enumerate(boxes.cls.cpu().numpy().astype(int)) if c == cat_class_id]

    if not cat_indices:
        raise ValueError("No cat detected.")

    # Use the first detected cat
    box = boxes.xyxy[cat_indices[0]].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box

    image = cv2.imread(image_path)
    cropped = image[y1:y2, x1:x2]

    resized = cv2.resize(cropped, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb
