"""
ai_model.siamese_model

Implements a Siamese network for cat re-identification using TensorFlow/Keras.
Based on the architecture from hsc-reident repository.

This module provides:
- Siamese network architecture with pre-trained base models
- Training functions with triplet loss
- Embedding extraction for inference
- Model saving and loading utilities
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from tensorflow.keras.models import Model
import numpy as np
import os
import pickle
from typing import List, Tuple, Optional
import cv2

class SiameseNetwork:
    def __init__(self, base_model_name: str = 'efficientnetb0', embedding_dim: int = 128):
        """
        Initialize Siamese network for cat re-identification.
        
        Args:
            base_model_name: Name of the base model ('efficientnetb0', 'vgg16', 'mobilenet')
            embedding_dim: Dimension of the embedding vector
        """
        self.base_model_name = base_model_name
        self.embedding_dim = embedding_dim
        self.model = None
        self.embedding_model = None
        self._build_model()
    
    def _get_base_model(self):
        """Get pre-trained base model."""
        if self.base_model_name == 'efficientnetb0':
            base_model = applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
        elif self.base_model_name == 'vgg16':
            base_model = applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
        elif self.base_model_name == 'mobilenet':
            base_model = applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")
        
        return base_model
    
    def _build_model(self):
        """Build the Siamese network architecture."""
        base_model = self._get_base_model()
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create embedding branch
        input_layer = layers.Input(shape=(224, 224, 3))
        x = layers.Rescaling(1./255)(input_layer)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        embedding = layers.Dense(self.embedding_dim, activation=None, name='embedding')(x)
        
        # Normalize embeddings
        embedding = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='normalized_embedding')(embedding)
        
        self.embedding_model = Model(input_layer, embedding, name='embedding_model')
        
        # Create Siamese model for training
        anchor_input = layers.Input(shape=(224, 224, 3), name='anchor_input')
        positive_input = layers.Input(shape=(224, 224, 3), name='positive_input')
        negative_input = layers.Input(shape=(224, 224, 3), name='negative_input')
        
        anchor_embedding = self.embedding_model(anchor_input)
        positive_embedding = self.embedding_model(positive_input)
        negative_embedding = self.embedding_model(negative_input)
        
        self.model = Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=[anchor_embedding, positive_embedding, negative_embedding],
            name='siamese_network'
        )
    
    def triplet_loss(self, margin=0.2):
        """Triplet loss function."""
        def loss(y_true, y_pred):
            anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
            
            # Calculate distances
            pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
            neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
            
            # Triplet loss
            basic_loss = pos_dist - neg_dist + margin
            loss = tf.maximum(basic_loss, 0.0)
            
            return tf.reduce_mean(loss)
        return loss
    
    def compile_model(self, learning_rate=0.0001):
        """Compile the model with triplet loss."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.triplet_loss())
    
    def get_embedding(self, image_array: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single image.
        
        Args:
            image_array: Image array of shape (H, W, 3) with values 0-255
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Ensure correct shape and type
        if image_array.shape[1:] != (224, 224, 3):
            image_array = cv2.resize(image_array, (224, 224))
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
        
        image_array = image_array.astype(np.float32)
        
        embedding = self.embedding_model.predict(image_array, verbose=0)
        return embedding[0]  # Return first (and only) embedding
    
    def save_model(self, model_path: str):
        """Save the embedding model."""
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        self.embedding_model.save(model_path)
    
    def load_model(self, model_path: str):
        """Load a saved embedding model."""
        if os.path.exists(model_path):
            self.embedding_model = keras.models.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model not found at {model_path}, using default model")

def create_triplet_dataset(image_paths: List[str], labels: List[int], batch_size: int = 32):
    """
    Create triplet dataset for training.
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        batch_size: Batch size for training
    
    Returns:
        TensorFlow dataset yielding triplets
    """
    def load_and_preprocess_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        return img
    
    def create_triplets(anchor_path, positive_path, negative_path):
        anchor = load_and_preprocess_image(anchor_path)
        positive = load_and_preprocess_image(positive_path)
        negative = load_and_preprocess_image(negative_path)
        return anchor, positive, negative
    
    # Create triplets (simplified - in practice you'd want more sophisticated triplet mining)
    anchor_paths, positive_paths, negative_paths = [], [], []
    
    # Group images by label
    label_to_paths = {}
    for path, label in zip(image_paths, labels):
        if label not in label_to_paths:
            label_to_paths[label] = []
        label_to_paths[label].append(path)
    
    # Create triplets
    for anchor_label, anchor_paths_list in label_to_paths.items():
        if len(anchor_paths_list) < 2:
            continue
        
        # Find negative samples (different label)
        negative_labels = [l for l in label_to_paths.keys() if l != anchor_label]
        if not negative_labels:
            continue
        
        for anchor_path in anchor_paths_list:
            # Positive sample (same label, different image)
            positive_paths_list = [p for p in anchor_paths_list if p != anchor_path]
            if not positive_paths_list:
                continue
            
            positive_path = np.random.choice(positive_paths_list)
            
            # Negative sample (different label)
            negative_label = np.random.choice(negative_labels)
            negative_path = np.random.choice(label_to_paths[negative_label])
            
            anchor_paths.append(anchor_path)
            positive_paths.append(positive_path)
            negative_paths.append(negative_path)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((anchor_paths, positive_paths, negative_paths))
    dataset = dataset.map(create_triplets, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Global model instance
_siamese_model = None

def get_siamese_model() -> SiameseNetwork:
    """Get or create the global Siamese model instance."""
    global _siamese_model
    if _siamese_model is None:
        model_path = "ai_model/weights/siamese_model"
        _siamese_model = SiameseNetwork()
        _siamese_model.load_model(model_path)
    return _siamese_model 