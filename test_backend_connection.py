#!/usr/bin/env python3
"""
Test script to verify backend connection and SavedModel loading.
Run this on your Mac to test the exported models.
"""

import requests
import json
import os
import sys
from pathlib import Path

def test_backend_status(base_url="http://localhost:5002"):
    """Test if the backend server is running"""
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            print(f"✅ Backend server is running at {base_url}")
            return True
        else:
            print(f"❌ Backend server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to backend at {base_url}")
        print("   Make sure the server is running: python serve.py")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Backend connection timed out at {base_url}")
        return False
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
        return False

def test_model_loading():
    """Test if the SavedModel can be loaded"""
    try:
        from ai_model.siamese_model import get_siamese_model
        
        print("🔄 Testing SavedModel loading...")
        model = get_siamese_model()
        print("✅ SavedModel loaded successfully")
        
        # Test with a dummy image
        import numpy as np
        dummy_image = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        embedding = model.get_embedding(dummy_image)
        
        print(f"✅ Embedding generated successfully: shape {embedding.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading SavedModel: {e}")
        return False

def test_image_upload(base_url="http://localhost:5002", test_image_path="test_cat.jpg"):
    """Test image upload and identification"""
    if not os.path.exists(test_image_path):
        print(f"⚠️ Test image not found: {test_image_path}")
        print("   Create a test image or use an existing cat photo")
        return False
    
    try:
        print(f"🔄 Testing image upload to {base_url}...")
        
        with open(test_image_path, 'rb') as f:
            files = {'image': (test_image_path, f, 'image/jpeg')}
            response = requests.post(f"{base_url}/identify", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Image upload successful")
            print(f"   Result: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ Image upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing image upload: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Backend Connection Test")
    print("=" * 40)
    
    # Configuration
    base_url = "http://localhost:5002"  # Update this to your Mac's IP if needed
    
    print(f"Testing backend at: {base_url}")
    print()
    
    # Test 1: Backend status
    print("1. Testing backend status...")
    backend_ok = test_backend_status(base_url)
    print()
    
    # Test 2: Model loading
    print("2. Testing SavedModel loading...")
    model_ok = test_model_loading()
    print()
    
    # Test 3: Image upload (if backend is running)
    if backend_ok:
        print("3. Testing image upload...")
        upload_ok = test_image_upload(base_url)
        print()
    else:
        print("3. Skipping image upload test (backend not available)")
        upload_ok = False
        print()
    
    # Summary
    print("📋 Test Summary:")
    print(f"   Backend Status: {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"   Model Loading: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"   Image Upload:  {'✅ PASS' if upload_ok else '❌ FAIL'}")
    
    if backend_ok and model_ok:
        print("\n🎉 Backend is ready for Expo Go!")
        print("\n📱 Next steps:")
        print("1. Update PBL3Expo/config/api.ts with your Mac's IP address")
        print("2. Start Expo Go on your phone")
        print("3. Test the cat identification feature")
    else:
        print("\n🔧 Issues found. Please fix them before testing with Expo Go.")
        
        if not backend_ok:
            print("\n   To start the backend:")
            print("   python serve.py")
            
        if not model_ok:
            print("\n   To fix model loading:")
            print("   - Ensure SavedModel files are in the project directory")
            print("   - Check that TensorFlow/Keras is properly installed")

if __name__ == "__main__":
    main() 