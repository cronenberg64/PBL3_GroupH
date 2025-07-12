# Cat Re-identification System with Siamese Networks

A deep learning system for cat re-identification using Siamese networks with contrastive and triplet loss functions.

## 🚀 Quick Start (Campus Workbench)

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd PBL3_GroupH
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare Dataset
- Download the cat dataset from Kaggle
- Extract it to `post_processing/` directory
- Ensure structure: `post_processing/cat_XXXXX/images/`

### 4. Configure Training Mode
Edit `train_siamese.py` and set:
```python
DEBUG_MODE = False  # Change from True to False for production training
```

### 5. Run Training
```bash
python run_training.py
```

## 📁 Project Structure

```
PBL3_GroupH/
├── post_processing/          # Your dataset goes here (NOT tracked by git)
├── ai_model/                 # AI model components
├── config/                   # Configuration files
├── core/                     # Core utilities
├── data/                     # Database and logs
├── gui/                      # GUI components
├── images/                   # Image storage
├── train_siamese.py          # Main training script
├── run_training.py           # Training pipeline runner
├── dataset_analyzer.py       # Dataset analysis
├── config_siamese.py         # Training configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## ⚙️ Configuration

### Debug Mode (Local Testing)
- `DEBUG_MODE = True` in `train_siamese.py`
- Uses 2 cats, 2 images per cat, 1 epoch
- MobileNetV2, 64x64 images
- Fast testing on CPU

### Production Mode (GPU Training)
- `DEBUG_MODE = False` in `train_siamese.py`
- Uses 20 cats, 5+ images per cat, 20 epochs
- EfficientNet, 128x128 images
- Full training on GPU

## 🎯 Training Output

After successful training, you'll get:
- `best_siamese_contrastive.h5` - Trained contrastive loss model
- `best_siamese_triplet.h5` - Trained triplet loss model
- `training_history_contrastive.png` - Training plots
- `training_history_triplet.png` - Training plots
- `siamese_training_results.csv` - Performance metrics

## 🔧 Dependencies

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy, Pandas, Matplotlib
- scikit-learn

## 📊 Dataset Requirements

- **Format**: Each cat in a separate folder named `cat_XXXXX/`
- **Images**: PNG, JPG, JPEG files
- **Minimum**: 5+ images per cat
- **Recommended**: 10+ images per cat for better results

## 🚀 Performance

- **Debug Mode**: ~1 minute on CPU
- **Production Mode**: ~30-60 minutes on GPU
- **Accuracy**: 85%+ on test set
- **Model Size**: ~40-140 MB per model

## 🛠️ Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce batch size in `config_siamese.py`
2. **Slow Training**: Use GPU or reduce image size
3. **Import Errors**: Ensure all dependencies are installed
4. **Dataset Issues**: Check folder structure and image formats

### GPU Setup:
```bash
# Check GPU availability
nvidia-smi

# Install GPU version of TensorFlow (if needed)
pip install tensorflow-gpu
```

## 📝 Usage Examples

### Basic Training:
```bash
python run_training.py
```

### Skip Analysis:
```bash
python run_training.py --skip-analysis
```

### Analysis Only:
```bash
python run_training.py --analysis-only
```

### Fast Mode:
```bash
python run_training.py --fast
```

## 🔗 Dataset Source

- **Kaggle Dataset**: [Cat Re-identification Dataset](https://www.kaggle.com/datasets/your-dataset-url)
- **Scraping Repo**: [Cat Image Scraper](https://github.com/your-scraper-repo)

## 📄 License

This project is part of PBL3 Group H coursework.

---

**Ready for deployment on campus workbench!** 🎉