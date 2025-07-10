import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Data processing and ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0, VGG16, MobileNetV2

# Image processing
import cv2
from PIL import Image
import os
from tqdm import tqdm
import random
from itertools import combinations

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EMBEDDING_DIM = 128
MARGIN = 1.0
LEARNING_RATE = 0.001
EPOCHS = 50
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

class SiameseDataset:
    def __init__(self, dataset_path, img_size=IMG_SIZE):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.images = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self, max_cats=None, min_images_per_cat=5):
        """Load images from the organized dataset"""
        print("Loading dataset...")
        
        # Get all cat folders
        cat_folders = [f for f in os.listdir(self.dataset_path) 
                      if f.startswith('cat_') and os.path.isdir(os.path.join(self.dataset_path, f))]
        
        # Sort folders to ensure consistent ordering
        cat_folders.sort()
        
        # Limit number of cats if specified
        if max_cats:
            cat_folders = cat_folders[:max_cats]
        
        print(f"Found {len(cat_folders)} cat folders")
        
        for cat_folder in tqdm(cat_folders, desc="Loading cats"):
            cat_path = os.path.join(self.dataset_path, cat_folder)
            
            # Get all image files in the cat folder
            image_files = [f for f in os.listdir(cat_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Skip cats with too few images
            if len(image_files) < min_images_per_cat:
                print(f"Skipping {cat_folder} - only {len(image_files)} images")
                continue
            
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
        
        print(f"Loaded {len(self.images)} images from {len(np.unique(self.labels))} cats")
        print(f"Image shape: {self.images.shape}")
        
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

class SiameseModel:
    def __init__(self, input_shape, embedding_dim=EMBEDDING_DIM, base_model='efficientnet'):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.base_model_name = base_model
        
    def create_embedding_model(self):
        """Create the base embedding model"""
        if self.base_model_name == 'efficientnet':
            base_model = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'vgg':
            base_model = VGG16(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        elif self.base_model_name == 'mobilenet':
            base_model = MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Create embedding model
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.embedding_dim, activation=None)(x)
        
        embedding_model = Model(inputs, outputs, name='embedding_model')
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
        
        # Calculate distance
        distance = self._euclidean_distance([embedding_a, embedding_b])
        
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
        
        # Calculate triplet loss
        loss = self._triplet_loss([anchor_embedding, positive_embedding, negative_embedding])
        
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
        return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))
    
    def _contrastive_loss(self, y_true, y_pred, margin=MARGIN):
        """Contrastive loss function"""
        y_true = tf.cast(y_true, y_pred.dtype)
        squared_preds = tf.square(y_pred)
        squared_margin = tf.square(tf.maximum(margin - y_pred, 0))
        loss = tf.reduce_mean((1 - y_true) * squared_preds + y_true * squared_margin)
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
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                f'best_siamese_{self.loss_type}.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
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
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        # Get embedding model for evaluation
        embedding_model = self.model.layers[2]  # Assuming embedding model is the 3rd layer
        
        # Get embeddings for test images
        test_embeddings = embedding_model.predict(test_data)
        
        # Simple nearest neighbor classification
        predictions = []
        for i, embedding in enumerate(test_embeddings):
            distances = []
            for j, other_embedding in enumerate(test_embeddings):
                if i != j:
                    dist = np.linalg.norm(embedding - other_embedding)
                    distances.append((dist, test_labels[j]))
            
            # Find nearest neighbor
            distances.sort(key=lambda x: x[0])
            predictions.append(distances[0][1])
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average='weighted')
        recall = recall_score(test_labels, predictions, average='weighted')
        f1 = f1_score(test_labels, predictions, average='weighted')
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
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
        plt.show()

def main():
    """Main training function"""
    print("Starting Siamese Network Training")
    print("=" * 50)
    
    # Step 1: Data Preparation
    print("\nStep 1: Data Preparation")
    print("-" * 30)
    
    # Initialize dataset
    dataset = SiameseDataset('siamese_dataset', img_size=IMG_SIZE)
    
    # Load dataset (limit to 20 cats for faster training)
    images, labels = dataset.load_dataset(max_cats=20, min_images_per_cat=5)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=TEST_SPLIT, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SPLIT, stratify=y_temp, random_state=42
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Step 2: Model Architecture
    print("\nStep 2: Model Architecture")
    print("-" * 30)
    
    # Create model
    siamese_model = SiameseModel(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        embedding_dim=EMBEDDING_DIM,
        base_model='efficientnet'
    )
    
    # Step 3: Training Pipeline
    print("\nStep 3: Training Pipeline")
    print("-" * 30)
    
    # Train with contrastive loss
    print("\nTraining with Contrastive Loss:")
    contrastive_model = siamese_model.create_siamese_model(loss_type='contrastive')
    
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