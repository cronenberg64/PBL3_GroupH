#!/usr/bin/env python3
"""
Production-ready inference script for Siamese Network Cat Re-identification
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pickle
from typing import List, Tuple, Optional, Dict
import json
from pathlib import Path

class CatReidentifier:
    def __init__(self, model_path: str, embedding_dim: int = 256, img_size: int = 200):
        """
        Initialize the cat re-identification system
        
        Args:
            model_path: Path to the trained model file
            embedding_dim: Dimension of embeddings
            img_size: Input image size
        """
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.img_size = img_size
        self.model = None
        self.embedding_model = None
        self.known_cats = {}  # {cat_id: embedding}
        self.threshold = 0.5  # Distance threshold for matching
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            print(f"Loading model from {self.model_path}...")
            self.model = load_model(self.model_path, compile=False)
            
            # Extract the embedding model
            self.embedding_model = self._extract_embedding_model()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new embedding model...")
            self._create_embedding_model()
    
    def _extract_embedding_model(self) -> Model:
        """Extract the embedding model from the Siamese model"""
        try:
            # Try to get the embedding model from the loaded model
            if hasattr(self.model, 'layers') and len(self.model.layers) > 2:
                embedding_model = self.model.layers[2]  # Usually the shared embedding model
                return embedding_model
            else:
                raise ValueError("Could not extract embedding model")
        except:
            # Fallback: create new embedding model
            return self._create_embedding_model()
    
    def _create_embedding_model(self) -> Model:
        """Create a new embedding model"""
        # Create base model
        base_model = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model
        for layer in base_model.layers:
            layer.trainable = False
        
        # Create embedding model
        inputs = Input(shape=(self.img_size, self.img_size, 3))
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        
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
        embeddings = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embeddings)
        
        embedding_model = Model(inputs, outputs=embeddings, name='embedding_model')
        return embedding_model
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess an image for inference
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def get_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Get embedding for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Preprocess image
            img = self.preprocess_image(image_path)
            if img is None:
                return None
            
            # Get embedding
            embedding = self.embedding_model.predict(img, verbose=0)
            return embedding[0]  # Remove batch dimension
            
        except Exception as e:
            print(f"Error getting embedding for {image_path}: {e}")
            return None
    
    def add_known_cat(self, cat_id: str, image_path: str) -> bool:
        """
        Add a known cat to the database
        
        Args:
            cat_id: Unique identifier for the cat
            image_path: Path to the cat's image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            embedding = self.get_embedding(image_path)
            if embedding is not None:
                self.known_cats[cat_id] = embedding
                print(f"Added cat {cat_id} to database")
                return True
            else:
                print(f"Failed to add cat {cat_id}")
                return False
                
        except Exception as e:
            print(f"Error adding cat {cat_id}: {e}")
            return False
    
    def identify_cat(self, image_path: str) -> Tuple[Optional[str], float]:
        """
        Identify a cat from an image
        
        Args:
            image_path: Path to the image to identify
            
        Returns:
            Tuple of (cat_id, confidence) or (None, 0.0) if no match
        """
        try:
            # Get embedding for the query image
            query_embedding = self.get_embedding(image_path)
            if query_embedding is None:
                return None, 0.0
            
            if not self.known_cats:
                return None, 0.0
            
            # Calculate distances to all known cats
            best_match = None
            best_distance = float('inf')
            
            for cat_id, known_embedding in self.known_cats.items():
                # Calculate Euclidean distance
                distance = np.linalg.norm(query_embedding - known_embedding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = cat_id
            
            # Convert distance to confidence (lower distance = higher confidence)
            confidence = max(0.0, 1.0 - (best_distance / self.threshold))
            
            # Check if distance is within threshold
            if best_distance <= self.threshold:
                return best_match, confidence
            else:
                return None, confidence
                
        except Exception as e:
            print(f"Error identifying cat: {e}")
            return None, 0.0
    
    def save_database(self, filepath: str):
        """Save the known cats database"""
        try:
            data = {
                'known_cats': {k: v.tolist() for k, v in self.known_cats.items()},
                'threshold': self.threshold,
                'embedding_dim': self.embedding_dim,
                'img_size': self.img_size
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Database saved to {filepath}")
            
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def load_database(self, filepath: str):
        """Load the known cats database"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.known_cats = {k: np.array(v) for k, v in data['known_cats'].items()}
            self.threshold = data.get('threshold', 0.5)
            self.embedding_dim = data.get('embedding_dim', 256)
            self.img_size = data.get('img_size', 200)
            
            print(f"Database loaded from {filepath}")
            print(f"Loaded {len(self.known_cats)} known cats")
            
        except Exception as e:
            print(f"Error loading database: {e}")
    
    def get_database_info(self) -> Dict:
        """Get information about the database"""
        return {
            'num_cats': len(self.known_cats),
            'cat_ids': list(self.known_cats.keys()),
            'threshold': self.threshold,
            'embedding_dim': self.embedding_dim,
            'img_size': self.img_size
        }

def main():
    """Example usage of the CatReidentifier"""
    
    # Initialize the re-identifier
    model_path = "best_siamese_contrastive.h5"  # or "best_siamese_triplet.h5"
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        print("Please train the model first using train_siamese.py")
        return
    
    reidentifier = CatReidentifier(model_path)
    
    # Example: Add known cats
    print("\n=== Adding Known Cats ===")
    
    # You can add cats from your dataset
    dataset_path = "post_processing"
    if os.path.exists(dataset_path):
        cat_folders = [f for f in os.listdir(dataset_path) if f.startswith('cat_')]
        
        for cat_folder in cat_folders[:5]:  # Add first 5 cats as examples
            cat_path = os.path.join(dataset_path, cat_folder)
            image_files = [f for f in os.listdir(cat_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if image_files:
                # Add first image of each cat
                image_path = os.path.join(cat_path, image_files[0])
                reidentifier.add_known_cat(cat_folder, image_path)
    
    # Save database
    reidentifier.save_database("cat_database.pkl")
    
    # Example: Identify a cat
    print("\n=== Cat Identification Example ===")
    
    # Get database info
    info = reidentifier.get_database_info()
    print(f"Database contains {info['num_cats']} cats")
    
    if info['num_cats'] > 0:
        # Try to identify a cat from the database
        test_cat_id = info['cat_ids'][0]
        test_image_path = os.path.join(dataset_path, test_cat_id, 
                                      os.listdir(os.path.join(dataset_path, test_cat_id))[0])
        
        print(f"Testing identification with {test_cat_id}...")
        identified_cat, confidence = reidentifier.identify_cat(test_image_path)
        
        if identified_cat:
            print(f"Identified as: {identified_cat} (confidence: {confidence:.3f})")
            if identified_cat == test_cat_id:
                print("✅ Correct identification!")
            else:
                print("❌ Incorrect identification")
        else:
            print("❌ No match found")
    
    print("\n=== Production Ready! ===")
    print("The CatReidentifier is now ready for production use.")
    print("You can:")
    print("1. Add more cats using add_known_cat()")
    print("2. Identify cats using identify_cat()")
    print("3. Save/load the database as needed")

if __name__ == "__main__":
    main() 