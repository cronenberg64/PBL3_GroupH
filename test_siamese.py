"""
test_siamese.py

Test script to verify the Siamese network architecture works correctly.
This script tests the model loading, embedding extraction, and matching.
"""

import os
import numpy as np
import cv2
from ai_model.siamese_model import SiameseNetwork, get_siamese_model
from ai_model.detect import preprocess_image
from ai_model.embedder import get_embedding
from ai_model.matcher import match_embedding

def test_siamese_model():
    """Test the Siamese network model."""
    print("Testing Siamese network architecture...")
    
    try:
        # Test model creation
        print("1. Creating Siamese network...")
        model = SiameseNetwork(base_model_name='efficientnetb0', embedding_dim=128)
        print("   ✓ Model created successfully")
        
        # Test embedding extraction
        print("2. Testing embedding extraction...")
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        embedding = model.get_embedding(dummy_image)
        print(f"   ✓ Embedding extracted: shape {embedding.shape}")
        print(f"   ✓ Embedding norm: {np.linalg.norm(embedding):.4f}")
        
        # Test global model instance
        print("3. Testing global model instance...")
        global_model = get_siamese_model()
        embedding2 = global_model.get_embedding(dummy_image)
        print(f"   ✓ Global model works: shape {embedding2.shape}")
        
        print("✓ All Siamese network tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_integration():
    """Test the full integration with detection and matching."""
    print("\nTesting full integration...")
    
    try:
        # Check if we have test images
        test_image_path = "./images/known_cats/ryusei_1.png"
        if not os.path.exists(test_image_path):
            print(f"Test image not found: {test_image_path}")
            return False
        
        print("1. Testing image preprocessing...")
        cropped = preprocess_image(test_image_path)
        print(f"   ✓ Image preprocessed: shape {cropped.shape}")
        
        print("2. Testing embedding extraction...")
        embedding = get_embedding(cropped)
        print(f"   ✓ Embedding extracted: shape {embedding.shape}")
        
        print("3. Testing matching...")
        # Create dummy database
        dummy_db = [
            {"id": "test_cat", "embedding": embedding}
        ]
        result = match_embedding(embedding, dummy_db)
        print(f"   ✓ Matching result: {result}")
        
        print("✓ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Siamese Network Architecture Test ===\n")
    
    # Test Siamese model
    model_test = test_siamese_model()
    
    # Test integration
    integration_test = test_integration()
    
    print("\n=== Test Summary ===")
    print(f"Model test: {'✓ PASSED' if model_test else '✗ FAILED'}")
    print(f"Integration test: {'✓ PASSED' if integration_test else '✗ FAILED'}")
    
    if model_test and integration_test:
        print("\n🎉 All tests passed! The Siamese network architecture is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 