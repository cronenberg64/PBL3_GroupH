# Siamese Network Training Pipeline

## Dataset Source

This project uses a high-quality cat re-identification dataset, originally scraped and organized for machine learning research:

- **Kaggle Dataset:** [Cat Re-Identification Image Dataset](https://www.kaggle.com/datasets/cronenberg64/cat-re-identification-image-dataset)
- **Scraping Toolkit:** [WebScrape_neko-jirushi GitHub Repository](https://github.com/cronenberg64/WebScrape_neko-jirushi)

The Kaggle dataset provides a ready-to-use, ML-friendly structure with thousands of cat images and metadata. The scraping toolkit repository contains the full pipeline for scraping, cleaning, and organizing the data from the source website, ensuring reproducibility and data provenance.

---

# Siamese Network Training Pipeline

This repository contains a complete pipeline for training Siamese networks for cat identification using your organized dataset.

## Overview

The training pipeline implements:
1. **Data Preparation**: Loading and preprocessing your organized cat dataset
2. **Model Architecture**: Siamese networks with both contrastive and triplet loss
3. **Training Pipeline**: Complete training with validation and evaluation

## Files Structure

```
├── train_siamese.py          # Main training script
├── dataset_analyzer.py       # Dataset analysis and visualization
├── config_siamese.py         # Configuration parameters
├── run_training.py           # Simple runner script
├── requirements_siamese.txt  # Python dependencies
├── .gitignore                # Ignore rules for code/data
└── README.md                 # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_siamese.txt
```

### 2. Run the Complete Pipeline

```bash
python run_training.py
```

This will:
- Check dependencies
- Analyze your dataset
- Train Siamese networks with both contrastive and triplet loss
- Generate evaluation results and visualizations

### 3. Alternative: Run Individual Steps

#### Dataset Analysis Only
```bash
python run_training.py --analysis-only
```

#### Skip Dataset Analysis
```bash
python run_training.py --skip-analysis
```

## Configuration

Edit `config_siamese.py` to customize training parameters:

```python
# Image and Model Parameters
IMG_SIZE = 224                    # Input image size
BATCH_SIZE = 32                   # Training batch size
EMBEDDING_DIM = 128               # Embedding dimension
MARGIN = 1.0                      # Margin for contrastive loss

# Training Parameters
LEARNING_RATE = 0.001             # Learning rate
EPOCHS = 50                       # Number of training epochs
MAX_CATS = 20                     # Maximum cats to use (for faster training)
MIN_IMAGES_PER_CAT = 5            # Minimum images per cat

# Model Architecture
BASE_MODEL = 'efficientnet'       # Base model: 'efficientnet', 'vgg', 'mobilenet'
LOSS_TYPE = 'both'                # Loss type: 'contrastive', 'triplet', 'both'
```

## Dataset Structure

Your dataset should be organized as follows:

```
siamese_dataset/
├── cat_0001_うみ/
│   ├── image_001.png
│   ├── image_002.png
│   ├── ...
│   └── info.json (optional)
├── cat_0002_cat_226475/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── ...
```

## Model Architecture

### Siamese Network with Contrastive Loss
- Uses pairs of images (positive: same cat, negative: different cats)
- Learns to minimize distance for positive pairs and maximize for negative pairs
- Good for binary similarity learning

### Siamese Network with Triplet Loss
- Uses triplets (anchor, positive, negative)
- Learns embeddings where positive is closer to anchor than negative
- Often provides better discriminative features

### Base Models
- **EfficientNetB0**: Good balance of accuracy and speed
- **VGG16**: Classic architecture, good for transfer learning
- **MobileNetV2**: Lightweight, good for mobile deployment

## Training Process

1. **Data Loading**: Loads images from your organized dataset
2. **Preprocessing**: Resizes images to 224x224 and normalizes to [0,1]
3. **Pair/Triplet Generation**: Creates training pairs or triplets
4. **Model Training**: Trains with early stopping and learning rate reduction
5. **Evaluation**: Tests on held-out data using nearest neighbor classification

## Output Files

After training, you'll get:

### Models
- `best_siamese_contrastive.h5`: Best model trained with contrastive loss
- `best_siamese_triplet.h5`: Best model trained with triplet loss

### Results
- `siamese_training_results.csv`: Training metrics and evaluation results
- `training_history_contrastive.png`: Training curves for contrastive loss
- `training_history_triplet.png`: Training curves for triplet loss

### Dataset Analysis
- `dataset_analysis.png`: Visualizations of dataset statistics
- `dataset_analysis.csv`: Detailed dataset information
- `selected_cats_for_training.csv`: List of cats used for training

## Performance Metrics

The pipeline evaluates models using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for each class (weighted average)
- **Recall**: Recall for each class (weighted average)
- **F1-Score**: Harmonic mean of precision and recall

## Customization

### Adding Data Augmentation

To enable data augmentation, modify `config_siamese.py`:

```python
USE_AUGMENTATION = True
AUGMENTATION_TYPES = ['flip', 'rotate', 'noise']
```

### Using Different Base Models

Change the base model in `config_siamese.py`:

```python
BASE_MODEL = 'vgg'  # or 'mobilenet'
```

### Adjusting Training Parameters

Modify training parameters in `config_siamese.py`:

```python
EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `BATCH_SIZE` or `MAX_CATS`
2. **Slow Training**: Use fewer cats or a smaller base model
3. **Poor Performance**: Increase `EPOCHS` or adjust `LEARNING_RATE`

### GPU Usage

The pipeline automatically uses GPU if available. To force CPU usage:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## Next Steps

After training, you can:

1. **Use the trained models** in your cat identification application
2. **Fine-tune hyperparameters** based on results
3. **Add more data** to improve performance
4. **Implement online learning** for new cats

## Example Usage in Your Application

```python
import tensorflow as tf
from train_siamese import SiameseModel

# Load trained model
model = tf.keras.models.load_model('best_siamese_contrastive.h5')

# Get embedding model
embedding_model = model.layers[2]  # Assuming embedding model is 3rd layer

# Generate embeddings for new images
embeddings = embedding_model.predict(new_images)

# Compare embeddings using Euclidean distance
# Lower distance = more similar cats
```

## Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify your dataset structure
3. Review the configuration parameters
4. Check the generated logs and error messages 