# Siamese Network Training Configuration

# Image and Model Parameters (Optimized for RTX 2080 - Memory Safe)
IMG_SIZE = 200  # Reduced from 224 to save memory
BATCH_SIZE = 16  # Reduced to prevent memory issues
EMBEDDING_DIM = 128
MARGIN = 1.0

# Training Parameters
LEARNING_RATE = 0.001
EPOCHS = 50
VALIDATION_SPLIT = 0.15  # More training data
TEST_SPLIT = 0.15  # Balanced split

# Dataset Parameters (Optimized for RTX 2080 - Memory Safe)
MAX_CATS = 120  # Reduced to prevent memory issues
MIN_IMAGES_PER_CAT = 3  # Better quality threshold
NUM_PAIRS_PER_IMAGE = 2  # Number of positive/negative pairs per image
NUM_TRIPLETS_PER_IMAGE = 1  # Number of triplets per image

# Model Architecture (Optimized for RTX 2080 - Memory Safe)
BASE_MODEL = 'efficientnetb3'  # Better than B2, safer than B4
# EfficientNet family: 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4'
# ResNet family: 'resnet50', 'resnet101', 'resnet152'
# DenseNet family: 'densenet121', 'densenet169', 'densenet201'
# Inception family: 'inceptionv3', 'inceptionresnetv2'
# VGG family: 'vgg16', 'vgg19'
# MobileNet family: 'mobilenetv2', 'mobilenetv3small', 'mobilenetv3large'
LOSS_TYPE = 'both'  # Options: 'contrastive', 'triplet', 'both'

# Data Augmentation (optional)
USE_AUGMENTATION = False
AUGMENTATION_TYPES = ['flip', 'rotate', 'noise']

# File Paths
DATASET_PATH = 'post_processing'
MODEL_SAVE_PATH = 'models/'
RESULTS_SAVE_PATH = 'results/'

# GPU Configuration
USE_GPU = True
MIXED_PRECISION = False

# Random Seeds
RANDOM_SEED = 42 