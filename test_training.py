#!/usr/bin/env python3
"""
Simple test script to identify where training is getting stuck
"""

import os
import sys
import time
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2

def test_basic_imports():
    print("1. Testing basic imports...")
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        print("✓ Basic imports successful")
        return True
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        return False

def test_tensorflow():
    print("2. Testing TensorFlow...")
    try:
        print(f"TensorFlow version: {tf.__version__}")
        test_tensor = tf.constant([1, 2, 3])
        print(f"✓ TensorFlow test successful: {test_tensor}")
        return True
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        return False

def test_model_loading():
    print("3. Testing model loading...")
    try:
        print("Loading EfficientNetB0...")
        start_time = time.time()
        model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
        load_time = time.time() - start_time
        print(f"✓ EfficientNetB0 loaded successfully in {load_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"✗ EfficientNetB0 failed: {e}")
        try:
            print("Trying MobileNetV2...")
            start_time = time.time()
            model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
            load_time = time.time() - start_time
            print(f"✓ MobileNetV2 loaded successfully in {load_time:.2f} seconds")
            return True
        except Exception as e2:
            print(f"✗ MobileNetV2 also failed: {e2}")
            return False

def test_dataset_path():
    print("4. Testing dataset path...")
    dataset_path = 'post_processing'
    if os.path.exists(dataset_path):
        print(f"✓ Dataset path exists: {dataset_path}")
        # Check if it has cat folders
        cat_folders = [f for f in os.listdir(dataset_path) 
                      if f.startswith('cat_') and os.path.isdir(os.path.join(dataset_path, f))]
        print(f"✓ Found {len(cat_folders)} cat folders")
        return True
    else:
        print(f"✗ Dataset path does not exist: {dataset_path}")
        return False

def main():
    print("Training Debug Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_tensorflow,
        test_model_loading,
        test_dataset_path
    ]
    
    for test in tests:
        if not test():
            print(f"\n❌ Test failed! Stopping here.")
            return False
        print()
    
    print("✅ All tests passed! Training should work.")
    return True

if __name__ == "__main__":
    main() 