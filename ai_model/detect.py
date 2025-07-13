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
import numpy as np

# Use pre-trained general object detector
model = YOLO("yolov8n.pt")  # Downloads automatically if missing


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Detect and crop cat from image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the cropped image (width, height)
        
    Returns:
        Cropped and resized RGB image array, or None if no cat detected
    """
    try:
        results = model(image_path)

        if not results or len(results[0].boxes) == 0:
            print(f"No objects detected in {image_path}")
            return None

        # Filter for 'cat' class (class 15 in COCO)
        cat_class_id = 15
        boxes = results[0].boxes
        cat_indices = [i for i, c in enumerate(boxes.cls.cpu().numpy().astype(int)) if c == cat_class_id]

        if not cat_indices:
            print(f"No cat detected in {image_path}")
            return None

        # Use the first detected cat
        box = boxes.xyxy[cat_indices[0]].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        # Ensure valid bounding box
        if x1 >= x2 or y1 >= y2:
            print(f"Invalid bounding box in {image_path}")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image {image_path}")
            return None
            
        cropped = image[y1:y2, x1:x2]
        
        # Ensure cropped image has valid dimensions
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            print(f"Invalid cropped dimensions in {image_path}")
            return None

        resized = cv2.resize(cropped, target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb
        
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None
