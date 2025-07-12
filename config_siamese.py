# Siamese Network Training Configuration

# Image and Model Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EMBEDDING_DIM = 128
MARGIN = 1.0

# Training Parameters
LEARNING_RATE = 0.001
EPOCHS = 50
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.3  # Increased to ensure enough samples per class

# Dataset Parameters
MAX_CATS = 20  # Limit number of cats for faster training
MIN_IMAGES_PER_CAT = 5  # Minimum images required per cat
NUM_PAIRS_PER_IMAGE = 2  # Number of positive/negative pairs per image
NUM_TRIPLETS_PER_IMAGE = 1  # Number of triplets per image

# Model Architecture
BASE_MODEL = 'efficientnet'  # Options: 'efficientnet', 'vgg', 'mobilenet'
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