from flask import Flask, request, jsonify
import os
import pickle
from ai_model.detect import preprocess_image
from ai_model.embedder import get_embedding
from ai_model.matcher import match_embedding
import numpy as np

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
    <html>
    <head>
      <title>猫の再識別システム</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        body { background: #fff; font-family: 'Segoe UI', sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
        .container { max-width: 400px; width: 100%; margin: 0 auto; padding: 32px 16px; box-sizing: border-box; }
        .title { font-size: 2rem; font-weight: bold; color: #222; text-align: center; margin-bottom: 32px; }
        .image-box { width: 180px; height: 180px; border-radius: 24px; background: #f3f4f6; display: flex; align-items: center; justify-content: center; margin: 0 auto 24px auto; overflow: hidden; }
        .upload-btn { background: #facc15; border: none; border-radius: 16px; padding: 14px 32px; color: #222; font-weight: bold; font-size: 16px; margin-bottom: 16px; cursor: pointer; width: 100%; max-width: 320px; }
        .button-group { width: 100%; margin-top: 32px; display: flex; flex-direction: column; align-items: center; }
        .confirm-btn, .cancel-btn { background: #fde68a; border: none; border-radius: 16px; padding: 14px 0; color: #b45309; font-weight: bold; font-size: 16px; width: 70%; margin-bottom: 12px; cursor: pointer; }
        .cancel-btn { margin-bottom: 0; }
        input[type="file"] { display: none; }
        .file-label { display: flex; flex-direction: column; align-items: center; cursor: pointer; }
        .file-label span { color: #aaa; font-size: 14px; margin-top: 8px; }
        #preview { width: 180px; height: 180px; object-fit: cover; border-radius: 24px; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="title">Upload Cat Photo</div>
        <form id="uploadForm" method="POST" action="/identify" enctype="multipart/form-data">
          <label class="file-label">
            <div class="image-box" id="imageBox">
              <img id="preview" src="" style="display:none;" />
              <svg id="cameraIcon" xmlns="http://www.w3.org/2000/svg" width="48" height="48" fill="none" viewBox="0 0 24 24" stroke="#facc15" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M15.232 5.232l1.536 1.536A2 2 0 0 1 18 8.268V17a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V8.268a2 2 0 0 1 1.232-1.5l1.536-1.536A2 2 0 0 1 10.768 4h2.464a2 2 0 0 1 1.5.232z"/><circle cx="12" cy="13" r="3"/></svg>
            </div>
            <input type="file" name="image" id="fileInput" accept="image/*" onchange="showPreview(event)" />
            <span id="fileName">Select Photo</span>
          </label>
          <div class="button-group">
            <button type="submit" class="confirm-btn" id="confirmBtn">Confirm</button>
            <button type="button" class="cancel-btn" onclick="resetForm()">Cancel</button>
          </div>
        </form>
      </div>
      <script>
        function showPreview(event) {
          const file = event.target.files[0];
          const preview = document.getElementById('preview');
          const cameraIcon = document.getElementById('cameraIcon');
          const fileName = document.getElementById('fileName');
          if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
              preview.src = e.target.result;
              preview.style.display = 'block';
              cameraIcon.style.display = 'none';
            };
            reader.readAsDataURL(file);
            fileName.textContent = 'Change Photo';
          } else {
            preview.src = '';
            preview.style.display = 'none';
            cameraIcon.style.display = 'block';
            fileName.textContent = 'Select Photo';
          }
        }
        function resetForm() {
          document.getElementById('uploadForm').reset();
          document.getElementById('preview').src = '';
          document.getElementById('preview').style.display = 'none';
          document.getElementById('cameraIcon').style.display = 'block';
          document.getElementById('fileName').textContent = 'Select Photo';
        }
      </script>
    </body>
    </html>
    '''

@app.route("/identify", methods=["POST"])
def identify_cat():
    print('Request files:', request.files)
    print('Request form:', request.form)
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

        # Convert numpy.bool_ to native bool for JSON serialization
        if "match_found" in result:
            result["match_found"] = bool(result["match_found"])

        # 医療情報を付加（存在する場合）
        matched_id = result.get("matched_id")
        if result["match_found"] and matched_id in medical_info_db:
            result["medical_info"] = medical_info_db[matched_id]

        # Convert all boolean values in your result dictionary to native Python `bool` before returning them with `jsonify`.
        for k, v in result.items():
            if isinstance(v, (np.bool_,)):
                result[k] = bool(v)

        return jsonify(result)
    except Exception as e:
        return jsonify({'match_found': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
