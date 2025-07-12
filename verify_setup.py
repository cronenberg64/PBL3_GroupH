#!/usr/bin/env python3
"""
Verification script to test if the setup is working correctly
"""

import os
import sys
import importlib

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'tensorflow',
        'numpy', 
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn',
        'cv2',
        'PIL',
        'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def check_gpu():
    """Check GPU availability"""
    print("\n🖥️ Checking GPU availability...")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
            return True
        else:
            print("⚠️ No GPU detected - training will be slower on CPU")
            return True
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def check_directories():
    """Check if necessary directories exist"""
    print("\n📁 Checking directories...")
    
    required_dirs = ['post_processing', 'results', 'models']
    missing = []
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - MISSING")
            missing.append(dir_name)
    
    if missing:
        print(f"\nCreating missing directories...")
        for dir_name in missing:
            os.makedirs(dir_name, exist_ok=True)
            print(f"✅ Created {dir_name}/")
    
    return True

def check_config():
    """Check configuration files"""
    print("\n⚙️ Checking configuration...")
    
    config_files = ['train_siamese.py', 'config_siamese.py', 'run_training.py']
    missing = []
    
    for file_name in config_files:
        if os.path.exists(file_name):
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - MISSING")
            missing.append(file_name)
    
    if missing:
        print(f"❌ Missing configuration files: {', '.join(missing)}")
        return False
    
    # Check debug mode setting
    try:
        with open('train_siamese.py', 'r') as f:
            content = f.read()
            if 'DEBUG_MODE = True' in content:
                print("⚠️ DEBUG_MODE is True (for local testing)")
            elif 'DEBUG_MODE = False' in content:
                print("✅ DEBUG_MODE is False (for production)")
            else:
                print("⚠️ DEBUG_MODE setting not found")
    except Exception as e:
        print(f"❌ Error reading train_siamese.py: {e}")
        return False
    
    return True

def test_tensorflow():
    """Test TensorFlow functionality"""
    print("\n🧪 Testing TensorFlow...")
    try:
        import tensorflow as tf
        import numpy as np
        
        # Test basic operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = a + b
        print(f"✅ TensorFlow basic operations: {c.numpy()}")
        
        # Test model creation
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(5,)),
            tf.keras.layers.Dense(1)
        ])
        print("✅ TensorFlow model creation")
        
        return True
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🔍 Cat Re-identification System Setup Verification")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_gpu,
        check_directories,
        check_config,
        test_tensorflow
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All checks passed! Your setup is ready for training.")
        print("\n📋 Next steps:")
        print("1. Download your dataset to post_processing/ directory")
        print("2. Run: python run_training.py")
        print("3. Monitor training progress")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("- Run: pip install -r requirements.txt")
        print("- Check your Python version (need 3.8+)")
        print("- Verify your dataset is in post_processing/")
    
    return all_passed

if __name__ == "__main__":
    main() 