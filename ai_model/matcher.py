"""
ai_model.matcher

Performs vector similarity matching between a new cat image embedding and
a database of known embeddings using Euclidean distance.

Functions:
    - match_embedding(query_embedding, db_embeddings): Returns the closest match or none.
Constants:
    - THRESHOLD: Distance threshold for determining a match.
"""


import numpy as np

# Euclidean distance threshold for match
THRESHOLD = 0.15

def match_embedding(query_embedding, db_embeddings, threshold=0.15):
    min_dist = float("inf")
    matched_id = None

    for entry in db_embeddings:
        db_id = entry["id"]
        db_embedding = entry["embedding"]
        dist = np.linalg.norm(np.array(query_embedding) - np.array(db_embedding))
        if dist < min_dist:
            min_dist = dist
            matched_id = db_id

    return {
        "match_found": min_dist < threshold,
        "matched_id": matched_id if min_dist < threshold else None,
        "score": min_dist
    }
