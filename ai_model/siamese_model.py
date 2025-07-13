"""
ai_model.siamese_model

Implements a Siamese network for cat re-identification using TensorFlow/Keras.
Based on the architecture from hsc-reident repository and our trained contrastive model.

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
    def __init__(self, base_model_name: str = 'efficientnetb3', embedding_dim: int = 128):
        """
        Initialize Siamese network for cat re-identification.
        
        Args:
            base_model_name: Name of the base model ('efficientnetb3', 'vgg16', 'mobilenet')
            embedding_dim: Dimension of the embedding vector
        """
        self.base_model_name = base_model_name
        self.embedding_dim = embedding_dim
        self.model = None
        self.embedding_model = None
        self._build_model()
    
    def _get_base_model(self):
        """Get pre-trained base model."""
        if self.base_model_name == 'efficientnetb3':
            base_model = applications.EfficientNetB3(
                weights='imagenet',
                include_top=False,
                input_shape=(200, 200, 3)  # Match our training configuration
            )
        elif self.base_model_name == 'efficientnetb0':
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
        """Build the Siamese network architecture matching our training."""
        base_model = self._get_base_model()
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create embedding branch matching our training architecture
        input_layer = layers.Input(shape=(200, 200, 3) if self.base_model_name == 'efficientnetb3' else (224, 224, 3))
        x = layers.Rescaling(1./255)(input_layer)
        x = base_model(x, training=False)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        embedding = layers.Dense(self.embedding_dim, activation=None, name='embeddings')(x)
        
        # Normalize embeddings (L2 normalization)
        embedding = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='normalized_embedding')(embedding)
        
        self.embedding_model = Model(input_layer, embedding, name='embedding_model')
        
        # Create Siamese model for training (contrastive loss architecture)
        input_a = layers.Input(shape=(200, 200, 3) if self.base_model_name == 'efficientnetb3' else (224, 224, 3))
        input_b = layers.Input(shape=(200, 200, 3) if self.base_model_name == 'efficientnetb3' else (224, 224, 3))
        
        embedding_a = self.embedding_model(input_a)
        embedding_b = self.embedding_model(input_b)
        
        # Calculate Euclidean distance
        distance = layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True)))([embedding_a, embedding_b])
        
        self.model = Model(
            inputs=[input_a, input_b],
            outputs=distance,
            name='siamese_network'
        )
    
    def contrastive_loss(self, margin=1.0):
        """Contrastive loss function matching our training."""
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, y_pred.dtype)
            squared_preds = tf.square(y_pred)
            squared_margin = tf.square(tf.maximum(margin - y_pred, 0))
            loss = tf.reduce_mean((1.0 - y_true) * squared_preds + y_true * squared_margin)
            return loss
        return loss
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with contrastive loss."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.contrastive_loss())
    
    def get_embedding(self, image_array: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single image.
        
        Args:
            image_array: Image array of shape (H, W, 3) with values 0-255
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        try:
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            
            # Ensure correct shape and type
            target_shape = (200, 200, 3) if self.base_model_name == 'efficientnetb3' else (224, 224, 3)
            
            # Check if resize is needed
            if image_array.shape[1:] != target_shape:
                # Ensure we have valid dimensions for resize
                if image_array.shape[1] > 0 and image_array.shape[2] > 0:
                    image_array = cv2.resize(image_array[0], (target_shape[0], target_shape[1]))
                    if len(image_array.shape) == 3:
                        image_array = np.expand_dims(image_array, axis=0)
                else:
                    # Create a default image if dimensions are invalid
                    image_array = np.zeros((1, target_shape[0], target_shape[1], 3), dtype=np.uint8)
            
            image_array = image_array.astype(np.float32)
            
            embedding = self.embedding_model.predict(image_array, verbose=0)
            return embedding[0]  # Return first (and only) embedding
            
        except Exception as e:
            print(f"Error in get_embedding: {e}")
            # Return a default embedding
            return np.zeros(self.embedding_dim)
    
    def save_model(self, model_path: str):
        """Save the embedding model."""
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        self.embedding_model.save(model_path)
    
    def load_model(self, model_path: str):
        """Load a saved embedding model."""
        if os.path.exists(model_path):
            try:
                # Try to load the full model first
                self.embedding_model = keras.models.load_model(model_path)
                print(f"Loaded model from {model_path}")
            except:
                # If that fails, try to load just the embedding model
                try:
                    # Load the full Siamese model and extract embedding model
                    full_model = keras.models.load_model(model_path, custom_objects={
                        'contrastive_loss': self.contrastive_loss()
                    })
                    # Extract the embedding model from the full model
                    self.embedding_model = full_model.layers[2]  # The shared embedding model
                    print(f"Loaded embedding model from full model at {model_path}")
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
                    print("Using default model architecture")
        else:
            print(f"Model not found at {model_path}, using default model")

# Global model instance
_siamese_model = None

def get_siamese_model() -> SiameseNetwork:
    """Get or create the global Siamese model instance."""
    global _siamese_model
    if _siamese_model is None:
        # Try to load our trained contrastive model
        model_path = "best_siamese_contrastive.h5"
        _siamese_model = SiameseNetwork(base_model_name='efficientnetb3', embedding_dim=128)
        _siamese_model.load_model(model_path)
    return _siamese_model 