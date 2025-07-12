# Campus Workbench Deployment Guide

This guide will help you deploy the Cat Re-identification System on your campus workbench with GPU support.

## 🚀 Quick Deployment

### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone <your-repo-url>
cd PBL3_GroupH

# Run the automated setup script
./setup_workbench.sh
```

### Option 2: Manual Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd PBL3_GroupH

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p post_processing results models

# Enable production mode
sed -i 's/DEBUG_MODE = True/DEBUG_MODE = False/' train_siamese.py
```

## 📊 System Requirements

### Minimum Requirements:
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 10GB+ free space
- **Python**: 3.8+

### Recommended (for GPU training):
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **RAM**: 16GB+
- **Storage**: 20GB+ free space

## 🔧 GPU Setup

### Check GPU Availability:
```bash
nvidia-smi
```

### Install GPU TensorFlow (if needed):
```bash
pip install tensorflow-gpu
```

### Verify GPU Access:
```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

## 📁 Dataset Preparation

1. **Download Dataset**: Get the cat dataset from Kaggle
2. **Extract**: Unzip to `post_processing/` directory
3. **Verify Structure**:
   ```
   post_processing/
   ├── cat_0001_うみ/
   │   ├── image_001.png
   │   └── ...
   ├── cat_0002_cat_226475/
   │   ├── image_001.png
   │   └── ...
   └── ...
   ```

## 🎯 Training Configuration

### Production Settings (Default):
- **Cats**: 20 (maximum)
- **Images per cat**: 5+ (minimum)
- **Epochs**: 20
- **Image size**: 128x128
- **Batch size**: 16
- **Base model**: EfficientNetB0

### Customization:
Edit `config_siamese.py` to adjust:
```python
EPOCHS = 50              # More epochs for better accuracy
BATCH_SIZE = 32          # Larger batch size for GPU
IMG_SIZE = 224           # Larger images for better accuracy
```

## 🚀 Running Training

### Basic Training:
```bash
python run_training.py
```

### Monitor Progress:
```bash
# In another terminal
watch -n 5 nvidia-smi  # Monitor GPU usage
```

### Expected Timeline:
- **Data loading**: 1-2 minutes
- **Model creation**: 2-3 minutes
- **Training**: 30-60 minutes (depending on GPU)
- **Total**: ~1 hour

## 📊 Expected Results

After successful training, you'll find:

### Models:
- `best_siamese_contrastive.h5` (~40MB)
- `best_siamese_triplet.h5` (~140MB)

### Visualizations:
- `training_history_contrastive.png`
- `training_history_triplet.png`

### Metrics:
- `siamese_training_results.csv`
- Expected accuracy: 85%+

## 🛠️ Troubleshooting

### Common Issues:

1. **Out of Memory**:
   ```bash
   # Reduce batch size in config_siamese.py
   BATCH_SIZE = 8
   ```

2. **Slow Training**:
   ```bash
   # Check GPU usage
   nvidia-smi
   
   # Verify TensorFlow GPU access
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

3. **Import Errors**:
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

4. **Dataset Issues**:
   ```bash
   # Check dataset structure
   ls -la post_processing/
   find post_processing/ -name "*.png" | wc -l
   ```

### Performance Optimization:

1. **Use SSD storage** for faster data loading
2. **Close other applications** to free GPU memory
3. **Monitor system resources** during training
4. **Use screen/tmux** for long-running sessions

## 📝 Monitoring Commands

### GPU Monitoring:
```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### System Monitoring:
```bash
# CPU and memory usage
htop

# Disk usage
df -h

# Process monitoring
ps aux | grep python
```

## 🔄 Switching Between Modes

### Debug Mode (for testing):
```bash
sed -i 's/DEBUG_MODE = False/DEBUG_MODE = True/' train_siamese.py
```

### Production Mode (for training):
```bash
sed -i 's/DEBUG_MODE = True/DEBUG_MODE = False/' train_siamese.py
```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your system meets the requirements
3. Check GPU availability and TensorFlow installation
4. Review the training logs for specific error messages

---

**Happy Training! 🎉** 