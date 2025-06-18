# test_identify.py
import pickle
from ai_model import identify_cat


# 識別前にデータベースを読み込む
with open("cat_embeddings.pkl", "rb") as f:
    database_embeddings = pickle.load(f)

result = identify_cat("./images/uploaded_cats/ryusei_6.png", database_embeddings)
print(result)