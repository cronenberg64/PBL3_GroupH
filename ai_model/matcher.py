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

def match_embedding(query_embedding, db_embeddings):
    if not db_embeddings:
        return {"match_found": False, "matched_id": None, "score": None}

    best_score = float("inf")
    best_id = None

    for entry_id, emb in db_embeddings.items():
        dist = np.linalg.norm(query_embedding - emb)
        if dist < best_score:
            best_score = dist
            best_id = entry_id

    if best_score < THRESHOLD:
        return {"match_found": True, "matched_id": best_id, "score": best_score}
    else:
        return {"match_found": False, "matched_id": None, "score": best_score}
