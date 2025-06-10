import os
import pickle
from detect import preprocess_image
from embedder import get_embedding

# 登録したい猫画像が入ったフォルダ
known_cats_folder = "./images/known_cats"  # 例: cat1.png, cat2.png, etc.

# 登録用リスト
db_embeddings = []

# フォルダ内の画像を読み込む
for filename in os.listdir(known_cats_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_cats_folder, filename)
        try:
            print(f"Processing {filename}...")
            cropped_img = preprocess_image(image_path)
            emb = get_embedding(cropped_img)
            db_embeddings.append({
                "id": filename.split('.')[0],  # "cat1" など
                "embedding": emb
            })
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# 保存
with open("cat_embeddings.pkl", "wb") as f:
    pickle.dump(db_embeddings, f)

print(" 登録完了: cat_embeddings.pkl に保存されました。")
