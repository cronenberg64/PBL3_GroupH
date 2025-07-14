"""
ai_model.register_known_cats

Registers known cats by creating embeddings from their images using our trained Siamese network.
This script processes images from the post_processing directory and creates a database of embeddings.

Functions:
    - register_known_cats(): Creates embeddings for all known cats
    - save_embeddings(): Saves embeddings to pickle file
"""

import os
import pickle
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from .embedder import get_embedding
from .detect import preprocess_image

def register_known_cats(dataset_path="post_processing", output_file="cat_embeddings.pkl"):
    """
    Register known cats by creating embeddings from their images.
    
    Args:
        dataset_path: Path to the dataset directory
        output_file: Output pickle file for embeddings
    
    Returns:
        List of embeddings with cat IDs
    """
    print("Starting cat registration with trained contrastive model...")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        return []
    
    embeddings_db = []
    cat_folders = [f for f in os.listdir(dataset_path) if f.startswith('cat_')]
    
    print(f"Found {len(cat_folders)} cat folders")
    
    for cat_folder in tqdm(cat_folders, desc="Processing cats"):
        cat_path = os.path.join(dataset_path, cat_folder)
        
        if not os.path.isdir(cat_path):
            continue
        
        # Get all image files in the cat folder
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend([f for f in os.listdir(cat_path) if f.lower().endswith(ext.split('*')[1])])
        
        if not image_files:
            print(f"No images found in {cat_folder}")
            continue
        
        print(f"Processing {cat_folder} with {len(image_files)} images")
        
        # Create embeddings for each image
        for img_file in image_files:
            img_path = os.path.join(cat_path, img_file)
            
            try:
                # Preprocess image (detect and crop cat)
                cropped_image = preprocess_image(img_path)
                
                if cropped_image is None:
                    print(f"Could not detect cat in {img_path}")
                    continue
                
                # Get embedding using our trained model
                embedding = get_embedding(cropped_image)
                
                # Create entry
                entry = {
                    "id": cat_folder,
                    "image_file": img_file,
                    "embedding": embedding,
                    "image_path": img_path
                }
                
                embeddings_db.append(entry)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    print(f"Successfully created {len(embeddings_db)} embeddings")
    
    # Save embeddings
    save_embeddings(embeddings_db, output_file)
    
    return embeddings_db

def save_embeddings(embeddings_db, output_file="cat_embeddings.pkl"):
    """
    Save embeddings to pickle file.
    
    Args:
        embeddings_db: List of embedding dictionaries
        output_file: Output file path
    """
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings_db, f)
        print(f"Saved {len(embeddings_db)} embeddings to {output_file}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")

def load_embeddings(input_file="cat_embeddings.pkl"):
    """
    Load embeddings from pickle file.
    
    Args:
        input_file: Input file path
    
    Returns:
        List of embeddings
    """
    try:
        with open(input_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded {len(embeddings)} embeddings from {input_file}")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return []

def get_cat_info(cat_id, embeddings_db):
    """
    Get information about a specific cat.
    
    Args:
        cat_id: Cat ID to look up
        embeddings_db: Database of embeddings
    
    Returns:
        Dictionary with cat information
    """
    cat_embeddings = [e for e in embeddings_db if e["id"] == cat_id]
    
    if not cat_embeddings:
        return None
    
    return {
        "id": cat_id,
        "num_images": len(cat_embeddings),
        "image_files": [e["image_file"] for e in cat_embeddings],
        "image_paths": [e["image_path"] for e in cat_embeddings]
    }

if __name__ == "__main__":
    # Register known cats
    embeddings = register_known_cats()
    
    if embeddings:
        print("\nRegistration Summary:")
        unique_cats = set(e["id"] for e in embeddings)
        print(f"Total unique cats: {len(unique_cats)}")
        print(f"Total embeddings: {len(embeddings)}")
        
        # Show some statistics
        for cat_id in list(unique_cats)[:5]:  # Show first 5 cats
            info = get_cat_info(cat_id, embeddings)
            if info:
                print(f"  {cat_id}: {info['num_images']} images")
    else:
        print("No embeddings created. Check dataset path and model availability.")
