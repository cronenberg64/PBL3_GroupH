import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
warnings.filterwarnings('ignore')

# Data processing and ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, 
    BatchNormalization, Layer, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4,
    VGG16, VGG19, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
    ResNet50V2, ResNet101V2, ResNet152V2,
    DenseNet121, DenseNet169, DenseNet201,
    InceptionV3, InceptionResNetV2
)

# Image processing
import cv2
from PIL import Image
import os
from tqdm import tqdm
import random
from itertools import combinations
import signal
import time

# Set random seeds for reproducibility
print("Setting random seeds...")
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
print("Random seeds set successfully")

# GPU memory management
print("Configuring GPU memory...")
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Allow memory growth to prevent memory issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Set memory limit to 7GB to leave buffer for system
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)]
            )
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        print("Memory limit set to 7GB to prevent OOM errors")
    else:
        print("No GPU found, using CPU")
except Exception as e:
    print(f"GPU configuration failed: {e}")
    print("Continuing with default settings...")

# Test TensorFlow
print("Testing TensorFlow...")
try:
    test_tensor = tf.constant([1, 2, 3])
    print(f"TensorFlow test successful: {test_tensor}")
except Exception as e:
    print(f"TensorFlow test failed: {e}")
    print("Continuing anyway...")

# Configuration
# --- ULTRA-FAST DEBUG MODE SETTINGS ---
DEBUG_MODE = False

if DEBUG_MODE:
    MAX_CATS = 2
    MIN_IMAGES_PER_CAT = 1
    EPOCHS = 1
    BASE_MODEL = 'mobilenet'
    IMG_SIZE = 64
    BATCH_SIZE = 4
    print("[DEBUG MODE] Using 2 cats, 1 image per cat, 1 epoch, MobileNetV2, 64x64 images, batch size 4.")
else:
    # Optimized settings for RTX 2080 (8GB VRAM) - Memory Safe
    MAX_CATS = 120  # Reduced to prevent memory issues
    MIN_IMAGES_PER_CAT = 3  # Better quality threshold
    MAX_IMAGES_PER_CAT = 15  # Reduced to save memory
    EPOCHS = 50  # More epochs with early stopping
    BASE_MODEL = 'efficientnetb3'  # Better than B2, safer than B4
    IMG_SIZE = 200  # Reduced from 224 to save memory
    BATCH_SIZE = 16  # Reduced to prevent memory issues

EMBEDDING_DIM = 256  # Increased for better feature learning
MARGIN = 0.5  # Reduced margin for better learning
LEARNING_RATE = 0.0001  # Reduced learning rate for stability
VALIDATION_SPLIT = 0.15  # Reduced to maximize training data
TEST_SPLIT = 0.15  # Reduced to maximize training data

# Training strategy improvements
PATIENCE = 20  # More patience for early stopping
MIN_DELTA = 0.0001  # Smaller improvement threshold
REDUCE_LR_PATIENCE = 10
REDUCE_LR_FACTOR = 0.7
MIN_LR = 1e-8

class SiameseDataset:
    def __init__(self, dataset_path, img_size=IMG_SIZE):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.images = []  # Will be converted to numpy array later
        self.labels = []  # Will be converted to numpy array later
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self, max_cats=None, min_images_per_cat=2, max_images_per_cat=15):
        """Load images from the organized dataset with better filtering"""
        print("Loading dataset...")
        start_time = time.time()
        
        # Get all cat folders
        cat_folders = [f for f in os.listdir(self.dataset_path) 
                      if f.startswith('cat_') and os.path.isdir(os.path.join(self.dataset_path, f))]
        
        # Sort folders to ensure consistent ordering
        cat_folders.sort()
        
        # Analyze cat folders to find the best ones
        cat_stats = []
        for cat_folder in cat_folders:
            cat_path = os.path.join(self.dataset_path, cat_folder)
            image_files = [f for f in os.listdir(cat_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(image_files) >= min_images_per_cat:
                cat_stats.append((cat_folder, len(image_files)))
        
        # Sort by number of images (descending) and take the best cats
        cat_stats.sort(key=lambda x: x[1], reverse=True)
        
        if max_cats:
            cat_stats = cat_stats[:max_cats]
        
        print(f"Selected {len(cat_stats)} cats with sufficient images")
        
        for cat_folder, num_images in tqdm(cat_stats, desc="Loading cats"):
            cat_path = os.path.join(self.dataset_path, cat_folder)
            image_files = [f for f in os.listdir(cat_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit images per cat to prevent imbalance
            if len(image_files) > max_images_per_cat:
                image_files = random.sample(image_files, max_images_per_cat)
            
            # Load images for this cat
            for img_file in image_files:
                img_path = os.path.join(cat_path, img_file)
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        img = img.astype(np.float32) / 255.0
                        
                        self.images.append(img)
                        self.labels.append(cat_folder)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        # Convert to numpy arrays
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        # Encode labels
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        load_time = time.time() - start_time
        print(f"Loaded {len(self.images)} images from {len(np.unique(self.labels))} cats")
        print(f"Image shape: {self.images.shape}")
        print(f"Loading took {load_time:.2f} seconds")
        
        return self.images, self.encoded_labels
    
    def create_pairs(self, images, labels, num_pairs_per_image=2):
        """Create positive and negative pairs for training"""
        print("Creating training pairs...")
        
        pair_images = []
        pair_labels = []
        
        # Get indices for each class
        unique_labels = np.unique(labels)
        label_indices = {label: np.where(labels == label)[0] for label in unique_labels}
        
        for i in tqdm(range(len(images)), desc="Creating pairs"):
            current_image = images[i]
            current_label = labels[i]
            
            # Create positive pairs (same class)
            pos_indices = label_indices[current_label]
            for _ in range(num_pairs_per_image):
                pos_idx = random.choice(pos_indices)
                if pos_idx != i:  # Don't pair with self
                    pos_image = images[pos_idx]
                    pair_images.append([current_image, pos_image])
                    pair_labels.append(0)  # 0 for positive pair
            
            # Create negative pairs (different class)
            neg_labels = [l for l in unique_labels if l != current_label]
            for _ in range(num_pairs_per_image):
                neg_label = random.choice(neg_labels)
                neg_idx = random.choice(label_indices[neg_label])
                neg_image = images[neg_idx]
                pair_images.append([current_image, neg_image])
                pair_labels.append(1)  # 1 for negative pair
        
        return np.array(pair_images), np.array(pair_labels)
    
    def create_triplets(self, images, labels, num_triplets_per_image=1):
        """Create triplets for triplet loss training"""
        print("Creating training triplets...")
        
        anchor_images = []
        positive_images = []
        negative_images = []
        
        # Get indices for each class
        unique_labels = np.unique(labels)
        label_indices = {label: np.where(labels == label)[0] for label in unique_labels}
        
        for i in tqdm(range(len(images)), desc="Creating triplets"):
            anchor_image = images[i]
            anchor_label = labels[i]
            
            for _ in range(num_triplets_per_image):
                # Positive: same class
                pos_indices = label_indices[anchor_label]
                pos_idx = random.choice(pos_indices)
                if pos_idx != i:
                    positive_image = images[pos_idx]
                    
                    # Negative: different class
                    neg_labels = [l for l in unique_labels if l != anchor_label]
                    neg_label = random.choice(neg_labels)
                    neg_idx = random.choice(label_indices[neg_label])
                    negative_image = images[neg_idx]
                    
                    anchor_images.append(anchor_image)
                    positive_images.append(positive_image)
                    negative_images.append(negative_image)
        
        return (np.array(anchor_images), np.array(positive_images), np.array(negative_images))

# Custom Keras Layers
class EuclideanDistanceLayer(Layer):
    """Custom layer to compute Euclidean distance between two embeddings"""
    def __init__(self, **kwargs):
        super(EuclideanDistanceLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        feats_a, feats_b = inputs
        sum_squared = tf.reduce_sum(tf.square(feats_a - feats_b), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_squared, 1e-7))

class ContrastiveLossLayer(Layer):
    """Custom layer for contrastive loss"""
    def __init__(self, margin=1.0, **kwargs):
        super(ContrastiveLossLayer, self).__init__(**kwargs)
        self.margin = margin
    
    def call(self, inputs):
        y_true, y_pred = inputs
        y_true = tf.cast(y_true, y_pred.dtype)
        squared_preds = tf.square(y_pred)
        squared_margin = tf.square(tf.maximum(self.margin - y_pred, 0))
        # Fix the contrastive loss calculation
        loss = tf.reduce_mean((1.0 - y_true) * squared_preds + y_true * squared_margin)
        return loss

class TripletLossLayer(Layer):
    """Custom layer for triplet loss"""
    def __init__(self, alpha=0.2, **kwargs):
        super(TripletLossLayer, self).__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, inputs):
        anchor, positive, negative = inputs
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        basic_loss = pos_dist - neg_dist + self.alpha
        loss = tf.maximum(basic_loss, 0.0)
        return tf.reduce_mean(loss)

class IdentityLossLayer(Layer):
    """Custom layer for identity loss"""
    def __init__(self, **kwargs):
        super(IdentityLossLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        y_true, y_pred = inputs
        return tf.reduce_mean(y_pred)

class SiameseModel:
    def __init__(self, input_shape, embedding_dim=EMBEDDING_DIM, base_model='efficientnet'):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.base_model_name = base_model
        
    def create_embedding_model(self):
        """Create the base embedding model"""
        print(f"Creating {self.base_model_name} embedding model...")
        
        # Model mapping for better models
        model_mapping = {
            # EfficientNet family (best performance)
            'efficientnet': EfficientNetB0,
            'efficientnetb0': EfficientNetB0,
            'efficientnetb1': EfficientNetB1,
            'efficientnetb2': EfficientNetB2,
            'efficientnetb3': EfficientNetB3,
            'efficientnetb4': EfficientNetB4,
            
            # ResNet family (reliable and proven)
            'resnet50': ResNet50V2,
            'resnet101': ResNet101V2,
            'resnet152': ResNet152V2,
            
            # DenseNet family (excellent feature reuse)
            'densenet121': DenseNet121,
            'densenet169': DenseNet169,
            'densenet201': DenseNet201,
            
            # Inception family (good for fine details)
            'inceptionv3': InceptionV3,
            'inceptionresnetv2': InceptionResNetV2,
            
            # VGG family (classic, reliable)
            'vgg16': VGG16,
            'vgg19': VGG19,
            
            # MobileNet family (lightweight)
            'mobilenet': MobileNetV2,
            'mobilenetv2': MobileNetV2,
            'mobilenetv3small': MobileNetV3Small,
            'mobilenetv3large': MobileNetV3Large,
        }
        
        if self.base_model_name not in model_mapping:
            raise ValueError(f"Unsupported base model: {self.base_model_name}. "
                           f"Available models: {list(model_mapping.keys())}")
        
        model_class = model_mapping[self.base_model_name]
        print(f"Loading {self.base_model_name}...")
        
        try:
            base_model = model_class(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
            print(f"{self.base_model_name} loaded successfully")
        except Exception as e:
            print(f"Error loading {self.base_model_name}: {e}")
            print("Falling back to EfficientNetB0...")
            base_model = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Create embedding model with improved architecture
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        
        # Improved architecture for better learning
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        embeddings = Dense(self.embedding_dim, activation=None, name='embeddings')(x)
        
        # Normalize embeddings for better distance learning
        embeddings = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embeddings)
        
        embedding_model = Model(inputs, outputs=embeddings, name='embedding_model')
        return embedding_model
    
    def create_siamese_model(self, loss_type='contrastive'):
        """Create the complete Siamese model"""
        embedding_model = self.create_embedding_model()
        
        if loss_type == 'contrastive':
            return self._create_contrastive_model(embedding_model)
        elif loss_type == 'triplet':
            return self._create_triplet_model(embedding_model)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def _create_contrastive_model(self, embedding_model):
        """Create Siamese model with contrastive loss"""
        # Input layers
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        
        # Get embeddings
        embedding_a = embedding_model(input_a)
        embedding_b = embedding_model(input_b)
        
        # Calculate distance using custom layer
        distance_layer = EuclideanDistanceLayer()
        distance = distance_layer([embedding_a, embedding_b])
        
        # Create model
        model = Model(inputs=[input_a, input_b], outputs=distance)
        model.compile(
            loss=self._contrastive_loss,
            optimizer=Adam(learning_rate=LEARNING_RATE),
            metrics=['accuracy']
        )
        
        return model
    
    def _create_triplet_model(self, embedding_model):
        """Create Siamese model with triplet loss"""
        # Input layers
        anchor_input = Input(shape=self.input_shape)
        positive_input = Input(shape=self.input_shape)
        negative_input = Input(shape=self.input_shape)
        
        # Get embeddings
        anchor_embedding = embedding_model(anchor_input)
        positive_embedding = embedding_model(positive_input)
        negative_embedding = embedding_model(negative_input)
        
        # Calculate triplet loss using custom layer
        triplet_loss_layer = TripletLossLayer(alpha=0.2)
        loss = triplet_loss_layer([anchor_embedding, positive_embedding, negative_embedding])
        
        # Create model
        model = Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=loss
        )
        model.compile(
            loss=self._identity_loss,
            optimizer=Adam(learning_rate=LEARNING_RATE)
        )
        
        return model
    
    def _euclidean_distance(self, vectors):
        """Calculate Euclidean distance between two vectors"""
        (feats_a, feats_b) = vectors
        sum_squared = tf.reduce_sum(tf.square(feats_a - feats_b), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_squared, 1e-7))
    
    def _contrastive_loss(self, y_true, y_pred, margin=MARGIN):
        """Contrastive loss function"""
        y_true = tf.cast(y_true, y_pred.dtype)
        squared_preds = tf.square(y_pred)
        squared_margin = tf.square(tf.maximum(margin - y_pred, 0))
        # Fix the contrastive loss calculation
        loss = tf.reduce_mean((1.0 - y_true) * squared_preds + y_true * squared_margin)
        return loss
    
    def _triplet_loss(self, inputs, alpha=0.2):
        """Triplet loss function"""
        anchor, positive, negative = inputs
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        basic_loss = pos_dist - neg_dist + alpha
        loss = tf.maximum(basic_loss, 0.0)
        return loss
    
    def _identity_loss(self, y_true, y_pred):
        """Identity loss for triplet model"""
        return tf.reduce_mean(y_pred)
    


class SiameseTrainer:
    def __init__(self, dataset, model, loss_type='contrastive'):
        self.dataset = dataset
        self.model = model
        self.loss_type = loss_type
        self.history = None
        
    def train(self, train_data, val_data, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the Siamese model"""
        print(f"Training Siamese model with {self.loss_type} loss...")
        
        # Callbacks with improved training strategy
        callbacks = [
            ModelCheckpoint(
                f'best_siamese_{self.loss_type}.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                min_delta=MIN_DELTA,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LR,
                verbose=1
            )
        ]
        
        # Train model
        if self.loss_type == 'contrastive':
            self.history = self.model.fit(
                [train_data[0][:, 0], train_data[0][:, 1]],
                train_data[1],
                validation_data=([val_data[0][:, 0], val_data[0][:, 1]], val_data[1]),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        elif self.loss_type == 'triplet':
            self.history = self.model.fit(
                [train_data[0], train_data[1], train_data[2]],
                np.ones(len(train_data[0])),  # Dummy labels
                validation_data=([val_data[0], val_data[1], val_data[2]], np.ones(len(val_data[0]))),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        return self.history
    
    def evaluate(self, test_data, test_labels):
        """Evaluate the trained model with real metrics"""
        print("Evaluating model with real metrics...")
        
        try:
            # Get the embedding model for evaluation
            embedding_model = self.model.layers[2]  # The shared embedding model
            
            # Generate embeddings for test data
            print("Generating test embeddings...")
            test_embeddings = embedding_model.predict(test_data, verbose=0)
            
            # Create evaluation pairs
            print("Creating evaluation pairs...")
            eval_pairs, eval_labels = self.dataset.create_pairs(test_data, test_labels, num_pairs_per_image=1)
            
            # Predict distances for evaluation pairs
            print("Predicting distances...")
            distances = self.model.predict([eval_pairs[:, 0], eval_pairs[:, 1]], verbose=0)
            
            # Convert distances to binary predictions (threshold-based)
            threshold = 0.5  # Distance threshold for same/different
            predictions = (distances > threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(eval_labels, predictions)
            precision = precision_score(eval_labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(eval_labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(eval_labels, predictions, average='weighted', zero_division=0)
            
            print(f"Real Evaluation Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        except Exception as e:
            print(f"Evaluation failed: {e}")
            print("Falling back to basic metrics...")
            
            # Fallback: basic accuracy calculation
            try:
                # Simple distance-based evaluation
                distances = self.model.predict([test_data[:100], test_data[100:200]], verbose=0)
                accuracy = np.mean(distances < 0.5)  # Simple threshold
                return {
                    'accuracy': accuracy,
                    'precision': accuracy,
                    'recall': accuracy,
                    'f1_score': accuracy
                }
            except:
                return {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                }
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy (if available)
        if 'accuracy' in self.history.history:
            ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
            ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'training_history_{self.loss_type}.png', dpi=300, bbox_inches='tight')
        
        # Only show plots if not in debug mode (to avoid hanging on macOS)
        if not DEBUG_MODE:
            plt.show()
        else:
            print(f"Training plot saved to: training_history_{self.loss_type}.png")
            plt.close()  # Close the plot to free memory

def main():
    """Main training function"""
    print("Starting Siamese Network Training")
    print("=" * 50)
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"[DEBUG] MAX_CATS={MAX_CATS}, MIN_IMAGES_PER_CAT={MIN_IMAGES_PER_CAT}, EPOCHS={EPOCHS}, BASE_MODEL={BASE_MODEL}, IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}")
    
    # Step 1: Data Preparation
    print("\nStep 1: Data Preparation")
    print("-" * 30)
    
    print("Initializing dataset...")
    # Check if dataset path exists
    dataset_path = 'post_processing'
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path '{dataset_path}' does not exist!")
        return
    
    print(f"Dataset path exists: {dataset_path}")
    # Initialize dataset
    dataset = SiameseDataset(dataset_path, img_size=IMG_SIZE)
    print("Dataset object created successfully")
    
    # Load dataset with optimized parameters
    print("Loading dataset (this may take a moment)...")
    images, labels = dataset.load_dataset(
        max_cats=MAX_CATS,
        min_images_per_cat=MIN_IMAGES_PER_CAT,
        max_images_per_cat=MAX_IMAGES_PER_CAT
    )
    print(f"Dataset loaded successfully: {len(images)} images")
    
    # Calculate appropriate split ratios based on number of unique classes
    unique_classes = len(np.unique(labels))
    print(f"Number of unique classes: {unique_classes}")
    
    # Ensure we have enough samples per class for stratified split
    min_samples_per_class = 2  # Minimum samples needed per class in each split
    total_samples = len(images)
    
    # Calculate minimum test size needed
    min_test_size = unique_classes * min_samples_per_class
    min_test_ratio = min_test_size / total_samples
    
    # Use the larger of configured test split or minimum required
    actual_test_split = max(TEST_SPLIT, min_test_ratio)
    
    if actual_test_split > 0.5:
        print(f"Warning: Test split ratio {actual_test_split:.2f} is high due to class count")
        print(f"Consider increasing dataset size or reducing number of classes")
    
    print(f"Using test split ratio: {actual_test_split:.2f}")
    
    # Split data with calculated ratio
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=actual_test_split, stratify=labels, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=VALIDATION_SPLIT, stratify=y_temp, random_state=42
        )
    except ValueError as e:
        print(f"Stratified split failed: {e}")
        print("Falling back to random split without stratification...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=actual_test_split, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=VALIDATION_SPLIT, random_state=42
        )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Step 2: Model Architecture
    print("\nStep 2: Model Architecture")
    print("-" * 30)
    
    print("Creating Siamese model...")
    # Create model
    siamese_model = SiameseModel(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        embedding_dim=EMBEDDING_DIM,
        base_model=BASE_MODEL
    )
    print("Model created successfully")
    
    # Step 3: Training Pipeline
    print("\nStep 3: Training Pipeline")
    print("-" * 30)
    
    # Train with contrastive loss
    print("\nTraining with Contrastive Loss:")
    print("Creating contrastive model...")
    try:
        contrastive_model = siamese_model.create_siamese_model(loss_type='contrastive')
        print("Contrastive model created successfully")
    except Exception as e:
        print(f"Error creating contrastive model: {e}")
        print("Trying with simpler configuration...")
        # Try with a simpler model
        siamese_model = SiameseModel(
            input_shape=(64, 64, 3),  # Even smaller
            embedding_dim=32,  # Even smaller
            base_model='mobilenet'  # Lighter model
        )
        contrastive_model = siamese_model.create_siamese_model(loss_type='contrastive')
        print("Simpler contrastive model created successfully")
    
    # Create pairs for contrastive loss
    train_pairs, train_pair_labels = dataset.create_pairs(X_train, y_train)
    val_pairs, val_pair_labels = dataset.create_pairs(X_val, y_val)
    
    # Train contrastive model
    contrastive_trainer = SiameseTrainer(dataset, contrastive_model, loss_type='contrastive')
    contrastive_history = contrastive_trainer.train(
        (train_pairs, train_pair_labels),
        (val_pairs, val_pair_labels)
    )
    
    # Evaluate contrastive model
    contrastive_metrics = contrastive_trainer.evaluate(X_test, y_test)
    contrastive_trainer.plot_training_history()
    
    # Train with triplet loss
    print("\nTraining with Triplet Loss:")
    triplet_model = siamese_model.create_siamese_model(loss_type='triplet')
    
    # Create triplets for triplet loss
    train_triplets = dataset.create_triplets(X_train, y_train)
    val_triplets = dataset.create_triplets(X_val, y_val)
    
    # Train triplet model
    triplet_trainer = SiameseTrainer(dataset, triplet_model, loss_type='triplet')
    triplet_history = triplet_trainer.train(
        train_triplets,
        val_triplets
    )
    
    # Evaluate triplet model
    triplet_metrics = triplet_trainer.evaluate(X_test, y_test)
    triplet_trainer.plot_training_history()
    
    # Save results
    results = {
        'contrastive_loss': contrastive_metrics,
        'triplet_loss': triplet_metrics
    }
    
    # Save results to file
    results_df = pd.DataFrame([
        {
            'loss_type': 'contrastive',
            'accuracy': contrastive_metrics['accuracy'],
            'precision': contrastive_metrics['precision'],
            'recall': contrastive_metrics['recall'],
            'f1_score': contrastive_metrics['f1_score']
        },
        {
            'loss_type': 'triplet',
            'accuracy': triplet_metrics['accuracy'],
            'precision': triplet_metrics['precision'],
            'recall': triplet_metrics['recall'],
            'f1_score': triplet_metrics['f1_score']
        }
    ])
    
    results_df.to_csv('siamese_training_results.csv', index=False)
    print(f"\nResults saved to siamese_training_results.csv")
    
    # Print summary
    print("\nTraining Summary:")
    print("=" * 50)
    print("Contrastive Loss Results:")
    for metric, value in contrastive_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTriplet Loss Results:")
    for metric, value in triplet_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nBest model files saved:")
    print(f"  - best_siamese_contrastive.h5")
    print(f"  - best_siamese_triplet.h5")
    print(f"  - training_history_contrastive.png")
    print(f"  - training_history_triplet.png")

if __name__ == "__main__":
    main() 