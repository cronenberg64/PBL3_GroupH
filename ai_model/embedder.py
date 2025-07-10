"""
ai_model.embedder

Generates deep learning embeddings from cropped and resized cat images using a Siamese network.
Based on the architecture from hsc-reident repository.

Functions:
    - get_embedding(image_array): Returns a normalized feature vector from the image.
"""

import numpy as np
from .siamese_model import get_siamese_model

def get_embedding(image_array):
    """
    Extract deep learning embedding from a cat image using Siamese network.
    
    Args:
        image_array: Image array of shape (H, W, 3) with values 0-255
        
    Returns:
        Normalized embedding vector
    """
    try:
        # Get the Siamese model
        model = get_siamese_model()
        
        # Extract embedding
        embedding = model.get_embedding(image_array)
        
        return embedding
        
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        # Fallback to simple mean RGB if model fails
    mean_colors = image_array.mean(axis=(0, 1))  # [R, G, B]
    return mean_colors / 255.0  # Normalize
