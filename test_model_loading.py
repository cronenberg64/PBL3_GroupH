#!/usr/bin/env python3
"""
test_model_loading.py

Simple script to test if our trained contrastive model loads correctly.
"""

import numpy as np
import tensorflow as tf
from ai_model.siamese_model import get_siamese_model

def test_model():
    print("Testing model loading...")
    
    try:
        # Get the model
        model = get_siamese_model()
        print("✓ Model loaded successfully")
        
        # Test with a sample image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        print(f"✓ Test image shape: {test_image.shape}")
        
        # Get embedding
        embedding = model.get_embedding(test_image)
        print(f"✓ Embedding shape: {embedding.shape}")
        print(f"✓ Embedding range: {embedding.min():.4f} to {embedding.max():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model() 