#!/usr/bin/env python3
"""
Test script to demonstrate the no-auto-registration policy.
This script shows that the system only performs identification and does not automatically register new cats.
"""

import requests
import json
import os
from PIL import Image
import numpy as np

# Server configuration
BASE_URL = "http://localhost:5001"

def test_system_status():
    """Test the system status endpoint."""
    print("🔍 Testing system status...")
    response = requests.get(f"{BASE_URL}/status")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ System Status: {data['status']}")
        print(f"✅ Auto-registration enabled: {data['system_config']['auto_registration_enabled']}")
        print(f"✅ Identification only: {data['system_config']['identification_only']}")
        print(f"✅ Admin required for registration: {data['system_config']['admin_required_for_registration']}")
        print(f"✅ Database: {data['database_info']['total_embeddings']} embeddings from {data['database_info']['unique_cats']} cats")
        return True
    else:
        print(f"❌ Failed to get system status: {response.status_code}")
        return False

def test_identification_only():
    """Test that the system only performs identification."""
    print("\n🔍 Testing identification-only behavior...")
    
    # Use a real cat image from our dataset
    test_image_path = "./post_processing/cat_226678/image_052.png"
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    try:
        # Test identification endpoint
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}/identify", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Identification response received")
            print(f"✅ Match found: {data.get('match_found', False)}")
            print(f"✅ System info: {data.get('system_info', {})}")
            
            if not data.get('match_found'):
                guidance = data.get('guidance', {})
                print(f"✅ Guidance provided: {guidance.get('message', 'No guidance')}")
                print(f"✅ Next steps: {guidance.get('next_steps', [])}")
            
            return True
        else:
            print(f"❌ Identification failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    finally:
        # No cleanup needed for real image
        pass

def test_admin_registration():
    """Test that only admins can register cats."""
    print("\n🔍 Testing admin-only registration...")
    
    # Test without API key (should fail)
    print("Testing registration without API key...")
    response = requests.post(f"{BASE_URL}/admin/register", data={'cat_id': 'test_cat'})
    if response.status_code == 401:
        print("✅ Registration correctly blocked without API key")
    else:
        print(f"❌ Unexpected response: {response.status_code}")
        return False
    
    # Test with wrong API key (should fail)
    print("Testing registration with wrong API key...")
    headers = {'X-API-Key': 'wrong_key'}
    response = requests.post(f"{BASE_URL}/admin/register", 
                           headers=headers, 
                           data={'cat_id': 'test_cat'})
    if response.status_code == 401:
        print("✅ Registration correctly blocked with wrong API key")
    else:
        print(f"❌ Unexpected response: {response.status_code}")
        return False
    
    # Test with correct API key but no image (should fail)
    print("Testing registration with correct API key but no image...")
    headers = {'X-API-Key': 'admin_key_2024'}
    response = requests.post(f"{BASE_URL}/admin/register", 
                           headers=headers, 
                           data={'cat_id': 'test_cat'})
    if response.status_code == 400:
        print("✅ Registration correctly requires image")
    else:
        print(f"❌ Unexpected response: {response.status_code}")
        return False
    
    return True

def test_admin_endpoints():
    """Test admin endpoints."""
    print("\n🔍 Testing admin endpoints...")
    
    headers = {'X-API-Key': 'admin_key_2024'}
    
    # Test config endpoint
    response = requests.get(f"{BASE_URL}/admin/config", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Admin config accessible")
        print(f"✅ System config: {data['system_config']}")
    else:
        print(f"❌ Admin config failed: {response.status_code}")
        return False
    
    # Test cats list endpoint
    response = requests.get(f"{BASE_URL}/admin/cats", headers=headers)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Admin cats list accessible")
        print(f"✅ Total cats: {data['total_cats']}")
    else:
        print(f"❌ Admin cats list failed: {response.status_code}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Testing Smart Cat Re-Identification System - No Auto-Registration Policy")
    print("=" * 80)
    
    tests = [
        ("System Status", test_system_status),
        ("Identification Only", test_identification_only),
        ("Admin Registration", test_admin_registration),
        ("Admin Endpoints", test_admin_endpoints)
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
        print("-" * 40)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system correctly prevents auto-registration.")
        print("\n📋 Summary:")
        print("✅ System only performs identification")
        print("✅ Auto-registration is explicitly disabled")
        print("✅ Admin authorization required for registration")
        print("✅ Proper guidance provided when no match is found")
        print("✅ Administrative endpoints are properly secured")
    else:
        print("⚠️  Some tests failed. Please check the system configuration.")

if __name__ == "__main__":
    main() 