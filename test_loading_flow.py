#!/usr/bin/env python3
"""
Test script to verify the loading flow works correctly.
"""

import requests
import json
import os

# Server configuration
BASE_URL = "http://localhost:5002"

def test_loading_flow():
    """Test the complete loading flow."""
    print("🚀 Testing Loading Flow")
    print("=" * 50)
    
    # Test 1: Check if server is running
    print("1. Checking server status...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is running")
            print(f"✅ Database: {data['database_info']['total_embeddings']} embeddings")
            print(f"✅ Model: {data['model_info']['model_type']}")
        else:
            print(f"❌ Server error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Test 2: Test identification with a real cat image
    print("\n2. Testing identification flow...")
    test_image_path = "./post_processing/cat_226678/image_052.png"
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}/identify", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Identification successful")
            print(f"✅ Match found: {data.get('match_found', False)}")
            print(f"✅ Confidence: {data.get('confidence', 0):.2f}%")
            
            # Check if system info is included
            if 'system_info' in data:
                print(f"✅ System info included")
                print(f"   - Auto-registration: {data['system_info']['auto_registration_enabled']}")
                print(f"   - Identification only: {data['system_info']['identification_only']}")
            
            # Check if guidance is provided for no match
            if not data.get('match_found') and 'guidance' in data:
                print(f"✅ Guidance provided for no match")
                print(f"   - Message: {data['guidance']['message']}")
                print(f"   - Next steps: {len(data['guidance']['next_steps'])} steps")
            
            return True
        else:
            print(f"❌ Identification failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during identification: {e}")
        return False

def main():
    """Run the loading flow test."""
    if test_loading_flow():
        print("\n🎉 Loading flow test PASSED!")
        print("\n📱 Ready for Expo Go testing:")
        print("1. Open Expo Go on your phone")
        print("2. Scan the QR code from your terminal")
        print("3. Navigate to Upload tab")
        print("4. Select a cat image")
        print("5. Click 'Confirm'")
        print("6. Watch the loading animation with cat walking GIF")
        print("7. See the progress updates:")
        print("   - 'Uploading image to server...'")
        print("   - 'Detecting cat in image...'")
        print("   - 'Extracting features...'")
        print("   - 'Matching against database...'")
        print("   - 'Analysis complete!'")
        print("8. Wait for 'Results ready!' message")
        print("9. Automatically redirect to results page")
    else:
        print("\n❌ Loading flow test FAILED!")
        print("Please check the backend server and model loading.")

if __name__ == "__main__":
    main() 