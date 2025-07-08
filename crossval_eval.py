from sklearn.model_selection import KFold
from ai_model.embedder import get_embedding
from ai_model.matcher import match_embedding
from PIL import Image
import os
import numpy as np

# 猫IDごとに画像ファイルをまとめる
def load_dataset(dataset_path):
    dataset = []  # (image_path, cat_id)
    for cat_id in os.listdir(dataset_path):
        cat_dir = os.path.join(dataset_path, cat_id)
        for file in os.listdir(cat_dir):
            if file.endswith(('.jpg', '.png')):
                dataset.append((os.path.join(cat_dir, file), cat_id))
    return dataset

def cross_validate(dataset, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    dataset = np.array(dataset)

    accuracies = []

    for train_idx, test_idx in kf.split(dataset):
        known_db = {}  
        for idx in train_idx:
            path, cat_id = dataset[idx]
            image = Image.open(path).convert("RGB")
            emb = get_embedding(image)
            known_db.setdefault(cat_id, []).append(emb)

        
        correct = 0
        total = 0
        for idx in test_idx:
            path, true_id = dataset[idx]
            image = Image.open(path).convert("RGB")
            query_emb = get_embedding(image)

            
            pred_id, _ = match_embedding(query_emb, known_db)
            if pred_id == true_id:
                correct += 1
            total += 1

        acc = correct / total
        accuracies.append(acc)
        print(f"Fold accuracy: {acc:.2%}")

    print(f"\nAverage Top-1 Accuracy: {np.mean(accuracies):.2%}")

if __name__ == '__main__':
    dataset = load_dataset('./images')  
    cross_validate(dataset)
