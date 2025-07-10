"""
ai_model.train_siamese

Training script for the Siamese network for cat re-identification.
This script trains the model on a dataset of cat images organized by individual cats.

Usage:
    python ai_model/train_siamese.py --dataset_path /path/to/cat/dataset --epochs 100
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from typing import List, Tuple
import pickle
from siamese_model import SiameseNetwork, create_triplet_dataset

def load_dataset_from_directory(dataset_path: str) -> Tuple[List[str], List[int]]:
    """
    Load dataset from directory structure where each subdirectory is a cat.
    
    Expected structure:
    dataset_path/
    ├── cat_001/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── cat_002/
    │   ├── image1.jpg
    │   └── ...
    └── ...
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []
    label_to_id = {}
    current_label = 0
    
    for cat_dir in sorted(os.listdir(dataset_path)):
        cat_path = os.path.join(dataset_path, cat_dir)
        if not os.path.isdir(cat_path):
            continue
            
        # Assign numeric label to this cat
        if cat_dir not in label_to_id:
            label_to_id[cat_dir] = current_label
            current_label += 1
        
        cat_label = label_to_id[cat_dir]
        
        # Load all images for this cat
        for img_file in os.listdir(cat_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cat_path, img_file)
                image_paths.append(img_path)
                labels.append(cat_label)
    
    print(f"Loaded {len(image_paths)} images from {len(label_to_id)} cats")
    return image_paths, labels

def train_siamese_model(
    dataset_path: str,
    model_save_path: str = "ai_model/weights/siamese_model",
    base_model: str = "efficientnetb0",
    embedding_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    validation_split: float = 0.2
):
    """
    Train the Siamese network on cat dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        model_save_path: Path to save the trained model
        base_model: Base model architecture
        embedding_dim: Dimension of embedding vectors
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        validation_split: Fraction of data to use for validation
    """
    print("Loading dataset...")
    image_paths, labels = load_dataset_from_directory(dataset_path)
    
    if len(image_paths) == 0:
        print("No images found in dataset!")
        return
    
    # Split into train and validation
    num_val = int(len(image_paths) * validation_split)
    train_paths = image_paths[num_val:]
    train_labels = labels[num_val:]
    val_paths = image_paths[:num_val]
    val_labels = labels[:num_val]
    
    print(f"Training on {len(train_paths)} images, validating on {len(val_paths)} images")
    
    # Create datasets
    train_dataset = create_triplet_dataset(train_paths, train_labels, batch_size)
    val_dataset = create_triplet_dataset(val_paths, val_labels, batch_size)
    
    # Create and compile model
    print("Creating Siamese network...")
    model = SiameseNetwork(base_model_name=base_model, embedding_dim=embedding_dim)
    model.compile_model(learning_rate=learning_rate)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path + "_best",
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train the model
    print("Starting training...")
    history = model.model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    print(f"Saving model to {model_save_path}")
    model.save_model(model_save_path)
    
    # Save training history
    history_path = model_save_path + "_history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    print("Training completed!")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Siamese network for cat re-identification")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset directory")
    parser.add_argument("--model_save_path", default="ai_model/weights/siamese_model", help="Path to save trained model")
    parser.add_argument("--base_model", default="efficientnetb0", choices=["efficientnetb0", "vgg16", "mobilenet"], help="Base model architecture")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split")
    
    args = parser.parse_args()
    
    # Train the model
    train_siamese_model(
        dataset_path=args.dataset_path,
        model_save_path=args.model_save_path,
        base_model=args.base_model,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split
    )

if __name__ == "__main__":
    main() 