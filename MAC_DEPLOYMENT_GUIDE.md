# 🍎 Mac Deployment Guide for Cat Re-Identification System

This guide will help you set up the backend on your Mac to work with Expo Go using the exported SavedModel.

## 📋 Prerequisites

- Mac with Python 3.8+ installed
- TensorFlow 2.19+ and Keras 3.10+
- The exported SavedModel files (transferred from Linux)
- Expo Go app on your phone

## 🚀 Step-by-Step Setup

### 1. **Transfer Models to Mac**

#### Option A: Google Drive (Recommended)
```bash
# On Linux (already done)
tar -czf siamese_models.tar.gz best_siamese_*_embedding_savedmodel/ model_info.json

# Upload siamese_models.tar.gz to Google Drive
# Download on Mac and extract:
tar -xzf siamese_models.tar.gz
```

#### Option B: USB Drive
```bash
# Copy the folders to USB drive
cp -r best_siamese_*_embedding_savedmodel/ /path/to/usb/
cp model_info.json /path/to/usb/

# On Mac, copy from USB to project directory
cp -r /path/to/usb/best_siamese_*_embedding_savedmodel/ ./
cp /path/to/usb/model_info.json ./
```

### 2. **Install Dependencies**

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Additional packages for Mac
pip install requests pillow opencv-python
```

### 3. **Verify Model Files**

```bash
# Check that models are present
ls -la best_siamese_*_embedding_savedmodel/
cat model_info.json

# Expected output:
# - best_siamese_contrastive_embedding_savedmodel/ (129MB)
# - best_siamese_triplet_embedding_savedmodel/ (129MB)
# - model_info.json (657 bytes)
```

### 4. **Test Backend Setup**

```bash
# Run the test script
python test_backend_connection.py

# Expected output:
# ✅ Backend server is running at http://localhost:5002
# ✅ SavedModel loaded successfully
# ✅ Embedding generated successfully: shape (128,)
```

### 5. **Start the Backend Server**

```bash
# Start the Flask server
python serve.py

# Expected output:
# ✅ Using exported SavedModel
# ✅ Successfully loaded SavedModel from best_siamese_contrastive_embedding_savedmodel
# * Running on http://0.0.0.0:5002
```

### 6. **Update Frontend Configuration**

Edit `PBL3Expo/config/api.ts`:

```typescript
export const API_CONFIG = {
  // Update this to your Mac's IP address
  BASE_URL: 'http://YOUR_MAC_IP:5002', // e.g., 'http://192.168.1.100:5002'
  
  // ... rest of config
};
```

To find your Mac's IP:
```bash
# On Mac terminal
ifconfig | grep "inet " | grep -v 127.0.0.1
```

### 7. **Test with Expo Go**

```bash
# In the PBL3Expo directory
cd PBL3Expo
npm install
npx expo start

# Scan QR code with Expo Go app
```

## 🔧 Troubleshooting

### **Model Loading Issues**

```bash
# Check TensorFlow version
python -c "import tensorflow as tf; print(tf.__version__)"

# Should be 2.19.0 or higher
```

### **Backend Connection Issues**

```bash
# Check if server is running
curl http://localhost:5002/status

# Check firewall settings
# System Preferences > Security & Privacy > Firewall
```

### **Expo Go Connection Issues**

1. **Same Network**: Ensure phone and Mac are on same WiFi
2. **IP Address**: Use Mac's local IP, not localhost
3. **Port**: Ensure port 5002 is not blocked

### **Model Performance Issues**

```bash
# Test model performance
python -c "
from ai_model.siamese_model import get_siamese_model
import numpy as np
import time

model = get_siamese_model()
test_image = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)

start_time = time.time()
embedding = model.get_embedding(test_image)
end_time = time.time()

print(f'Embedding time: {(end_time - start_time)*1000:.2f}ms')
print(f'Embedding shape: {embedding.shape}')
"
```

## 📱 Testing the Complete System

### 1. **Backend Test**
```bash
# Test with a cat image
curl -X POST -F "image=@test_cat.jpg" http://localhost:5002/identify
```

### 2. **Frontend Test**
1. Open Expo Go app
2. Scan QR code from `npx expo start`
3. Navigate to Upload screen
4. Select a cat photo
5. Wait for identification results

### 3. **Expected Results**
- ✅ Backend loads SavedModel successfully
- ✅ Image upload works
- ✅ Cat detection and identification works
- ✅ Results display correctly in Expo Go

## 🎯 Performance Optimization

### **For Better Performance**

1. **Use SSD**: Ensure models are on SSD for faster loading
2. **Close Other Apps**: Free up memory for model inference
3. **Network**: Use 5GHz WiFi for faster image upload
4. **Image Size**: Optimize image size before upload

### **Memory Usage**
- SavedModel: ~129MB per model
- Runtime memory: ~500MB-1GB
- Total system memory needed: ~2GB free

## 🔒 Security Notes

1. **Local Network Only**: Backend runs on local network
2. **No External Access**: Server not exposed to internet
3. **Image Privacy**: Images processed locally, not stored permanently

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run `python test_backend_connection.py` for diagnostics
3. Check server logs in terminal
4. Verify network connectivity between devices

## ✅ Success Checklist

- [ ] Models transferred to Mac
- [ ] Dependencies installed
- [ ] Backend test passes
- [ ] Server starts successfully
- [ ] Frontend config updated with correct IP
- [ ] Expo Go connects to backend
- [ ] Image upload works
- [ ] Cat identification works

🎉 **Congratulations!** Your cat re-identification system is now ready for use with Expo Go! 