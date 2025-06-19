from flask import Flask, request, jsonify
import os
import pickle
from ai_model.detect import preprocess_image
from ai_model.embedder import get_embedding
from ai_model.matcher import match_embedding

app = Flask(__name__)

# 猫の埋め込み特徴量を読み込む
EMBEDDING_FILE = "cat_embeddings.pkl"
if os.path.exists(EMBEDDING_FILE):
    with open(EMBEDDING_FILE, "rb") as f:
        db_embeddings = pickle.load(f)
else:
    db_embeddings = []

# 仮の医療情報DB（本番はDB連携でもOK）
medical_info_db = {
    "cat_001": {
        "name": "ミケ",
        "gender": "female",
        "vaccinated": True,
        "last_visit": "2024-05-01"
    },
    "cat_002": {
        "name": "タマ",
        "gender": "male",
        "vaccinated": False,
        "last_visit": "2023-12-15"
    }
}

@app.route("/", methods=["GET"])
def upload_form():
    return '''
        <h2>猫の再識別システム</h2>
        <form method="POST" action="/identify" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" capture="environment"><br><br>
            <input type="submit" value="照合する">
        </form>
    '''

@app.route("/identify", methods=["POST"])
def identify_cat():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    temp_path = "temp_upload.png"
    image.save(temp_path)

    try:
        # 猫を検出して特徴量を取得
        cropped = preprocess_image(temp_path)
        embedding = get_embedding(cropped)
        result = match_embedding(embedding, db_embeddings)

        # 医療情報を付加（存在する場合）
        matched_id = result.get("matched_id")
        if result["match_found"] and matched_id in medical_info_db:
            result["medical_info"] = medical_info_db[matched_id]

        return jsonify(result)
    except Exception as e:
        return jsonify({'match_found': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
