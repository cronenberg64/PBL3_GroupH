"""
ai_model.register_known_cats

Script to register known cats in the database using the Siamese network.
This script processes images from the known_cats folder and creates embeddings
for the re-identification system.

Usage:
    python ai_model/register_known_cats.py
"""

import os
import pickle
from detect import preprocess_image
from embedder import get_embedding

def register_known_cats(known_cats_folder="./images/known_cats", output_file="cat_embeddings.pkl"):
    """
    Register known cats by processing their images and creating embeddings.
    
    Args:
        known_cats_folder: Path to folder containing cat images
        output_file: Path to save the embeddings database
    """
    # Database to store embeddings
db_embeddings = []

    if not os.path.exists(known_cats_folder):
        print(f"Known cats folder not found: {known_cats_folder}")
        return
    
    # Process each file in the known cats folder
for filename in os.listdir(known_cats_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(known_cats_folder, filename)
        try:
            print(f"Processing {filename}...")
                
                # Detect and crop cat from image
                cropped_img = preprocess_image(image_path)
                
                # Generate embedding using Siamese network
                embedding = get_embedding(cropped_img)
                
                # Store in database
                cat_id = filename.split('.')[0]  # Remove extension
                db_embeddings.append({
                    "id": cat_id,
                    "embedding": embedding,
                    "image_path": image_path
                })
                
                print(f"  ✓ Registered {cat_id}")
                
            except Exception as e:
                print(f"  ✗ Failed to process {filename}: {e}")
    
    # Save embeddings to file
    with open(output_file, "wb") as f:
        pickle.dump(db_embeddings, f)
    
    print(f"\nRegistration complete: {len(db_embeddings)} cats registered in {output_file}")
    return db_embeddings

def register_cats_from_subdirectories(base_folder="./images/known_cats", output_file="cat_embeddings.pkl"):
    """
    Register cats from a directory structure where each subdirectory is a cat.
    
    Expected structure:
    base_folder/
    ├── cat_001/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── cat_002/
    │   ├── image1.jpg
    │   └── ...
    └── ...
    
    Args:
        base_folder: Path to base folder containing cat subdirectories
        output_file: Path to save the embeddings database
    """
    db_embeddings = []
    
    if not os.path.exists(base_folder):
        print(f"Base folder not found: {base_folder}")
        return
    
    # Process each cat subdirectory
    for cat_dir in sorted(os.listdir(base_folder)):
        cat_path = os.path.join(base_folder, cat_dir)
        if not os.path.isdir(cat_path):
            continue
            
        print(f"Processing cat: {cat_dir}")
        
        # Process all images for this cat
        for filename in os.listdir(cat_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(cat_path, filename)
                try:
                    print(f"  Processing {filename}...")
                    
                    # Detect and crop cat from image
            cropped_img = preprocess_image(image_path)
                    
                    # Generate embedding using Siamese network
                    embedding = get_embedding(cropped_img)
                    
                    # Store in database
            db_embeddings.append({
                        "id": cat_dir,  # Use directory name as cat ID
                        "embedding": embedding,
                        "image_path": image_path
            })
                    
                    print(f"    ✓ Registered {cat_dir}")
                    
        except Exception as e:
                    print(f"    ✗ Failed to process {filename}: {e}")

    # Save embeddings to file
    with open(output_file, "wb") as f:
    pickle.dump(db_embeddings, f)

    print(f"\nRegistration complete: {len(db_embeddings)} embeddings registered in {output_file}")
    return db_embeddings

if __name__ == "__main__":
    # Check if we have subdirectories (organized structure) or flat structure
    known_cats_folder = "./images/known_cats"
    
    if os.path.exists(known_cats_folder):
        # Check if there are subdirectories
        has_subdirs = any(os.path.isdir(os.path.join(known_cats_folder, item)) 
                         for item in os.listdir(known_cats_folder))
        
        if has_subdirs:
            print("Detected organized directory structure. Registering cats from subdirectories...")
            register_cats_from_subdirectories(known_cats_folder)
        else:
            print("Detected flat directory structure. Registering cats from files...")
            register_known_cats(known_cats_folder)
    else:
        print(f"Known cats folder not found: {known_cats_folder}")
        print("Please create the folder and add cat images.")
