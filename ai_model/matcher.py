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

# Distance threshold for match (adjusted for our contrastive learning model)
# Based on our training results with 69.4% accuracy
THRESHOLD = 0.4  # Distance threshold from our training evaluation

def match_embedding(query_embedding, db_embeddings, threshold=0.4):
    """
    Match a query embedding against database embeddings.
    
    Args:
        query_embedding: Query embedding vector
        db_embeddings: List of database embeddings with 'id' and 'embedding' keys
        threshold: Distance threshold for determining a match (0.4 from our training)
        
    Returns:
        Dictionary with match_found, matched_id, score, and confidence
    """
    if not db_embeddings:
        return {
            "match_found": False,
            "matched_id": None,
            "score": float('inf'),
            "confidence": 0.0
        }
    
    min_dist = float("inf")
    matched_id = None
    best_similarity = 0.0
    
    query_embedding = np.array(query_embedding).flatten()

    for entry in db_embeddings:
        db_id = entry["id"]
        db_embedding = np.array(entry["embedding"]).flatten()
        
        # Calculate Euclidean distance (our model uses Euclidean distance)
        distance = np.sqrt(np.sum((query_embedding - db_embedding) ** 2))
        
        # Also calculate cosine similarity for confidence scoring
        similarity = cosine_similarity([query_embedding], [db_embedding])[0][0]
        
        if distance < min_dist:
            min_dist = distance
            matched_id = db_id
            best_similarity = similarity
    
    # Determine if it's a match based on threshold
    is_match = min_dist < threshold
    
    # Calculate confidence based on distance (closer = higher confidence)
    if is_match:
        # Normalize distance to confidence (0-1 scale)
        # Distance of 0 = 100% confidence, distance of threshold = 50% confidence
        confidence = max(0.5, 1.0 - (min_dist / threshold))
    else:
        confidence = 0.0

    return {
        "match_found": is_match,
        "matched_id": matched_id if is_match else None,
        "score": min_dist,  # Distance score (lower is better)
        "similarity": best_similarity,  # Similarity score (higher is better)
        "confidence": confidence,  # Confidence score (0-1)
        "threshold": threshold  # Threshold used for matching
    }
