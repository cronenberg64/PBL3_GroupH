#!/usr/bin/env python3
"""
test_integration.py

Test script to verify the integration of our trained contrastive learning model
with the frontend and backend systems.

This script:
1. Tests model loading
2. Tests embedding generation
3. Tests cat registration
4. Tests the complete identification pipeline
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_model.embedder import get_embedding
from ai_model.detect import preprocess_image
from ai_model.matcher import match_embedding
from ai_model.register_known_cats import register_known_cats, load_embeddings

def test_model_loading():
    """Test if our trained model loads correctly."""
    print("Testing model loading...")
    try:
        embedding = get_embedding(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        print(f"✓ Model loaded successfully")
        print(f"✓ Embedding shape: {embedding.shape}")
        print(f"✓ Embedding range: {embedding.min():.4f} to {embedding.max():.4f}")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_embedding_generation():
    """Test embedding generation with sample images."""
    print("\nTesting embedding generation...")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    try:
        embedding = get_embedding(test_image)
        print(f"✓ Embedding generated successfully")
        print(f"✓ Embedding dimension: {len(embedding)}")
        return True
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return False

def test_cat_registration():
    """Test cat registration from dataset."""
    print("\nTesting cat registration...")
    
    if not os.path.exists("post_processing"):
        print("✗ Dataset directory 'post_processing' not found")
        print("  Please ensure the dataset is available")
        return False
    
    try:
        embeddings = register_known_cats()
        if embeddings:
            print(f"✓ Cat registration successful")
            print(f"✓ Registered {len(embeddings)} embeddings")
            
            # Show some statistics
            unique_cats = set(e["id"] for e in embeddings)
            print(f"✓ Unique cats: {len(unique_cats)}")
            
            return True
        else:
            print("✗ No embeddings created during registration")
            return False
    except Exception as e:
        print(f"✗ Cat registration failed: {e}")
        return False

def test_matching_pipeline():
    """Test the complete matching pipeline."""
    print("\nTesting matching pipeline...")
    
    # Load embeddings
    embeddings = load_embeddings()
    if not embeddings:
        print("✗ No embeddings found. Run cat registration first.")
        return False
    
    # Create a test query embedding
    test_embedding = np.random.rand(128)  # 128-dimensional embedding
    test_embedding = test_embedding / np.linalg.norm(test_embedding)  # Normalize
    
    try:
        result = match_embedding(test_embedding, embeddings)
        print(f"✓ Matching pipeline successful")
        print(f"✓ Match found: {result['match_found']}")
        print(f"✓ Confidence: {result.get('confidence', 'N/A')}")
        print(f"✓ Score: {result.get('score', 'N/A')}")
        return True
    except Exception as e:
        print(f"✗ Matching pipeline failed: {e}")
        return False

def test_image_processing():
    """Test image processing pipeline."""
    print("\nTesting image processing...")
    
    # Create a test image file
    test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    test_path = "test_image.jpg"
    cv2.imwrite(test_path, test_image)
    
    try:
        # Test preprocessing
        cropped = preprocess_image(test_path)
        if cropped is not None:
            print(f"✓ Image preprocessing successful")
            print(f"✓ Cropped image shape: {cropped.shape}")
            
            # Test embedding generation
            embedding = get_embedding(cropped)
            print(f"✓ Embedding generation successful")
            print(f"✓ Embedding shape: {embedding.shape}")
            
            # Clean up
            os.remove(test_path)
            return True
        else:
            print("✗ Image preprocessing failed - no cat detected")
            os.remove(test_path)
            return False
    except Exception as e:
        print(f"✗ Image processing failed: {e}")
        if os.path.exists(test_path):
            os.remove(test_path)
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Smart Cat Re-Identification System - Integration Test")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Embedding Generation", test_embedding_generation),
        ("Image Processing", test_image_processing),
        ("Cat Registration", test_cat_registration),
        ("Matching Pipeline", test_matching_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for deployment.")
        print("\nNext steps:")
        print("1. Start the backend server: python serve.py")
        print("2. Start the mobile app: cd PBL3Expo && npm start")
        print("3. Test the complete system with real cat images")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure the trained model file exists: best_siamese_contrastive.h5")
        print("2. Check that the dataset is available in post_processing/")
        print("3. Verify all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 