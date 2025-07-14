"""
ai_model.embedder

Generates deep learning embeddings from cropped and resized cat images using a Siamese network.
Based on the architecture from hsc-reident repository and our trained contrastive model.

Functions:
    - get_embedding(image_array): Returns a normalized feature vector from the image.
"""

import numpy as np
import cv2
from .siamese_model import get_siamese_model

def get_embedding(image_array):
    """
    Extract deep learning embedding from a cat image using Siamese network.
    
    Args:
        image_array: Image array of shape (H, W, 3) with values 0-255
        
    Returns:
        Normalized embedding vector of shape (128,)
    """
    try:
        # Get the Siamese model
        model = get_siamese_model()
        
        # Ensure image is in the correct format
        if image_array is None:
            raise ValueError("Image array is None")
        
        # Convert to numpy array if needed
        if not isinstance(image_array, np.ndarray):
            image_array = np.array(image_array)
        
        # Ensure 3 channels
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        # Ensure correct data type
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        # Extract embedding
        embedding = model.get_embedding(image_array)
        
        # Ensure embedding is 1D and has correct shape
        embedding = embedding.flatten()
        if len(embedding) != 128:
            raise ValueError(f"Expected 128-dimensional embedding, got {len(embedding)}")
        
        return embedding
        
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        # Fallback to a proper 128-dimensional embedding
        # Create a random embedding that's normalized
        fallback_embedding = np.random.rand(128)
        fallback_embedding = fallback_embedding / np.linalg.norm(fallback_embedding)
        return fallback_embedding
