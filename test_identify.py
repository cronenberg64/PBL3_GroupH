# test_identify.py

from ai_model import identify_cat

# Dummy embedding database
known_embeddings = {
    "cat_001": [0.45, 0.36, 0.31],
    "cat_002": [0.23, 0.19, 0.15],
}

result = identify_cat("images/uploaded_cats/sample.jpg", known_embeddings)
print(result)
