"""
ai_model.embedder

Generates a simple image embedding from a cropped and resized cat image.

Currently uses a placeholder embedding strategy (mean RGB values) for simplicity.
Can be upgraded to use a deep learning feature extractor like ResNet or MobileNet.

Functions:
    - get_embedding(image_array): Returns a normalized feature vector from the image.
"""


import numpy as np

def get_embedding(image_array):
    # Fake embedding: mean RGB values flattened
    mean_colors = image_array.mean(axis=(0, 1))  # [R, G, B]
    return mean_colors / 255.0  # Normalize
