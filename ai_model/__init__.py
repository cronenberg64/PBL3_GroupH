"""
ai_model.__init__

Main entry point for the AI-based stray cat identification pipeline.

This module provides the `identify_cat` function, which takes an image path
and a database of known cat embeddings, and returns the best match based on
image analysis using YOLO detection and vector similarity.

Functions:
    - identify_cat(image_path, database_embeddings): Returns the closest cat match.
"""


from .detect import preprocess_image
from .embedder import get_embedding
from .matcher import match_embedding
from tqdm import tqdm

def identify_cat(image_path, database_embeddings):
    result = {}
    steps = ["Detecting cat", "Generating embedding", "Matching embedding"]

    with tqdm(total=len(steps), desc="Identifying cat", unit="step") as pbar:
        try:
            pbar.set_description("Step 1: Detecting cat")
            cat_image = preprocess_image(image_path)
            pbar.update(1)

            pbar.set_description("Step 2: Generating embedding")
            embedding = get_embedding(cat_image)
            pbar.update(1)

            pbar.set_description("Step 3: Matching embedding")
            result = match_embedding(embedding, database_embeddings)
            pbar.update(1)

        except Exception as e:
            result = {"match_found": False, "error": str(e)}

    return result