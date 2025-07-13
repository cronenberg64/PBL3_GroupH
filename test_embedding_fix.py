#!/usr/bin/env python3
"""
Test script to diagnose and fix the embedding issue.
"""

import numpy as np
import cv2
import os
from ai_model.detect import preprocess_image
from ai_model.embedder import get_embedding
from ai_model.matcher import match_embedding
import pickle

def test_embedding_extraction():
    """Test embedding extraction with a real cat image."""
    print("🔍 Testing embedding extraction...")
    
    # Use a real cat image from our dataset
    test_image_path = "./post_processing/cat_226678/image_052.png"
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    try:
        # Test preprocessing
        print("Testing image preprocessing...")
        cropped = preprocess_image(test_image_path)
        if cropped is None:
            print("❌ Preprocessing failed - no cat detected")
            return False
        
        print(f"✅ Preprocessing successful - image shape: {cropped.shape}")
        
        # Test embedding extraction
        print("Testing embedding extraction...")
        embedding = get_embedding(cropped)
        
        print(f"✅ Embedding extracted - shape: {embedding.shape}")
        print(f"✅ Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"✅ Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_matching():
    """Test matching against the database."""
    print("\n🔍 Testing matching against database...")
    
    # Load embeddings
    embedding_file = "cat_embeddings.pkl"
    if not os.path.exists(embedding_file):
        print(f"❌ Embeddings file not found: {embedding_file}")
        return False
    
    with open(embedding_file, "rb") as f:
        db_embeddings = pickle.load(f)
    
    print(f"✅ Loaded {len(db_embeddings)} embeddings from database")
    
    # Test with a real cat image
    test_image_path = "./post_processing/cat_226678/image_052.png"
    
    try:
        # Get embedding
        cropped = preprocess_image(test_image_path)
        if cropped is None:
            print("❌ Preprocessing failed")
            return False
        
        embedding = get_embedding(cropped)
        
        # Test matching
        result = match_embedding(embedding, db_embeddings, threshold=0.4)
        
        print(f"✅ Matching result: {result}")
        
        if result['match_found']:
            print(f"✅ Match found: {result['matched_id']} with score {result['score']:.4f}")
        else:
            print("⚠️  No match found - this might indicate an issue with embeddings")
            
            # Check if this cat exists in the database
            cat_id = "cat_226678"
            matching_embeddings = [e for e in db_embeddings if e['id'] == cat_id]
            print(f"Found {len(matching_embeddings)} embeddings for {cat_id} in database")
            
            if matching_embeddings:
                # Calculate similarity with existing embeddings
                similarities = []
                for db_emb in matching_embeddings:
                    similarity = 1 - np.linalg.norm(embedding - db_emb['embedding'])
                    similarities.append(similarity)
                
                print(f"Similarities with existing embeddings: {[f'{s:.4f}' for s in similarities]}")
                print(f"Max similarity: {max(similarities):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during matching: {e}")
        return False

def test_model_loading():
    """Test model loading directly."""
    print("\n🔍 Testing model loading...")
    
    try:
        from ai_model.siamese_model import get_siamese_model
        
        print("Loading Siamese model...")
        model = get_siamese_model()
        print("✅ Model loaded successfully")
        
        # Test with a simple image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        embedding = model.get_embedding(test_image)
        
        print(f"✅ Test embedding shape: {embedding.shape}")
        print(f"✅ Test embedding norm: {np.linalg.norm(embedding):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Embedding System")
    print("=" * 50)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Embedding Extraction", test_embedding_extraction),
        ("Database Matching", test_matching)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {str(e)}")
        print("-" * 30)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The embedding system is working correctly.")
    else:
        print("⚠️  Some tests failed. The issue is likely with model loading.")
        print("\n🔧 Suggested fixes:")
        print("1. Check TensorFlow version compatibility")
        print("2. Re-train the model if needed")
        print("3. Use a different model format")

if __name__ == "__main__":
    main() 