"""
ai_model.matcher

Performs vector similarity matching between a new cat image embedding and
a database of known embeddings using cosine similarity and Euclidean distance.

Functions:
    - match_embedding(query_embedding, db_embeddings): Returns the closest match or none.
Constants:
    - THRESHOLD: Distance threshold for determining a match.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Distance threshold for match (adjusted for deep learning embeddings)
THRESHOLD = 0.3  # Cosine distance threshold

def match_embedding(query_embedding, db_embeddings, threshold=0.3):
    """
    Match a query embedding against database embeddings.
    
    Args:
        query_embedding: Query embedding vector
        db_embeddings: List of database embeddings with 'id' and 'embedding' keys
        threshold: Distance threshold for determining a match
        
    Returns:
        Dictionary with match_found, matched_id, and score
    """
    if not db_embeddings:
        return {
            "match_found": False,
            "matched_id": None,
            "score": float('inf')
        }
    
    min_dist = float("inf")
    matched_id = None
    best_similarity = 0.0
    
    query_embedding = np.array(query_embedding).flatten()

    for entry in db_embeddings:
        db_id = entry["id"]
        db_embedding = np.array(entry["embedding"]).flatten()
        
        # Calculate cosine similarity (higher is better)
        similarity = cosine_similarity([query_embedding], [db_embedding])[0][0]
        
        # Convert to distance (lower is better)
        distance = 1 - similarity
        
        if distance < min_dist:
            min_dist = distance
            matched_id = db_id
            best_similarity = similarity
    
    # Determine if it's a match based on threshold
    is_match = min_dist < threshold

    return {
        "match_found": is_match,
        "matched_id": matched_id if is_match else None,
        "score": min_dist,  # Distance score (lower is better)
        "similarity": best_similarity  # Similarity score (higher is better)
    }
