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
        return jsonify(result)
    except Exception as e:
        return jsonify({'match_found': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # スマホと同じWi-Fiに接続していること前提
    app.run(host="0.0.0.0", port=5000, debug=True)
