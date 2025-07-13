from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
from ai_model.detect import preprocess_image
from ai_model.embedder import get_embedding
from ai_model.matcher import match_embedding
import tensorflow as tf
import json
from datetime import datetime

app = Flask(__name__)

# Load cat embeddings database
EMBEDDING_FILE = "cat_embeddings.pkl"
if os.path.exists(EMBEDDING_FILE):
    with open(EMBEDDING_FILE, "rb") as f:
        db_embeddings = pickle.load(f)
    print(f"Loaded {len(db_embeddings)} embeddings from {EMBEDDING_FILE}")
else:
    db_embeddings = []
    print(f"No embeddings file found at {EMBEDDING_FILE}")

# Medical information database (in production, this would be a real database)
medical_info_db = {
    "cat_226805": {
        "name": "ミケ",
        "gender": "female",
        "vaccinated": True,
        "last_visit": "2024-05-01",
        "status": "neutered"
    },
    "cat_226810": {
        "name": "タマ",
        "gender": "male",
        "vaccinated": False,
        "last_visit": "2023-12-15",
        "status": "under_treatment"
    },
    "cat_226815": {
        "name": "クロ",
        "gender": "male",
        "vaccinated": True,
        "last_visit": "2024-03-20",
        "status": "released"
    }
}

# System configuration
SYSTEM_CONFIG = {
    "auto_registration_enabled": False,  # Explicitly disabled
    "admin_required_for_registration": True,
    "max_upload_size_mb": 10,
    "supported_formats": ["jpg", "jpeg", "png", "gif"],
    "identification_only": True  # System only performs identification, not registration
}

# Store the last result for browser-based flows
last_result = None

@app.route("/", methods=["GET"])
def upload_form():
    return '''
    <html>
    <head>
      <title>Smart Cat Re-Identification System</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        body { background: #fff; font-family: 'Segoe UI', sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
        .container { max-width: 400px; width: 100%; margin: 0 auto; padding: 32px 16px; box-sizing: border-box; }
        .title { font-size: 2rem; font-weight: bold; color: #222; text-align: center; margin-bottom: 32px; }
        .subtitle { font-size: 1rem; color: #666; text-align: center; margin-bottom: 32px; }
        .image-box { width: 240px; height: 240px; border-radius: 32px; background: #f3f4f6; display: flex; align-items: center; justify-content: center; margin: 0 auto 24px auto; overflow: hidden; border: 2px solid #facc15; box-shadow: 0 4px 16px rgba(0,0,0,0.06); transition: box-shadow 0.2s; }
        .image-box:hover { box-shadow: 0 8px 32px rgba(0,0,0,0.10); }
        .upload-btn { background: #facc15; border: none; border-radius: 16px; padding: 14px 32px; color: #222; font-weight: bold; font-size: 16px; margin-bottom: 16px; cursor: pointer; width: 100%; max-width: 320px; }
        .button-group { width: 100%; margin-top: 32px; display: flex; flex-direction: column; align-items: center; }
        .confirm-btn, .cancel-btn { background: #fde68a; border: none; border-radius: 16px; padding: 14px 0; color: #b45309; font-weight: bold; font-size: 16px; width: 70%; margin-bottom: 12px; cursor: pointer; }
        .cancel-btn { margin-bottom: 0; }
        input[type="file"] { display: none; }
        .file-label { display: flex; flex-direction: column; align-items: center; cursor: pointer; }
        .file-label span { color: #aaa; font-size: 14px; margin-top: 8px; }
        #preview { width: 240px; height: 240px; object-fit: cover; border-radius: 32px; }
        .model-info { font-size: 0.8rem; color: #888; text-align: center; margin-top: 16px; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="title">Smart Cat Re-ID</div>
        <div class="subtitle">Upload a photo to identify a previously registered cat</div>
        <div style="font-size: 0.8rem; color: #666; text-align: center; margin-bottom: 16px; padding: 8px; background: #fef3c7; border-radius: 8px;">
          ℹ️ This system only identifies cats that have been previously registered by authorized personnel.
        </div>
        <form id="uploadForm" method="POST" action="/identify" enctype="multipart/form-data">
          <label class="file-label">
            <div class="image-box" id="imageBox">
              <img id="preview" src="" style="display:none;" />
              <svg id="cameraIcon" xmlns="http://www.w3.org/2000/svg" width="72" height="72" fill="none" viewBox="0 0 24 24" stroke="#facc15" stroke-width="2">
                <rect x="3" y="7" width="18" height="14" rx="4" stroke="#facc15" stroke-width="2" fill="#fff"/>
                <circle cx="12" cy="14" r="4" stroke="#facc15" stroke-width="2" fill="none"/>
                <rect x="8" y="3" width="8" height="4" rx="2" stroke="#facc15" stroke-width="2" fill="#fff"/>
              </svg>
            </div>
            <input type="file" name="image" id="fileInput" accept="image/*" onchange="showPreview(event)" />
            <span id="fileName">Select Photo</span>
          </label>
          <div class="button-group">
            <button type="submit" class="confirm-btn" id="confirmBtn">Identify Cat</button>
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
    global last_result
    print('Request files:', request.files)
    print('Request form:', request.form)
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    temp_path = "temp_upload.png"
    image.save(temp_path)

    # If the request is from a browser (not API), show a loading page first
    if 'text/html' in request.accept_mimetypes:
        # Start processing in the background (for demo, we process synchronously)
        # In production, use a task queue or async processing
        try:
            # Detect and crop cat from image
            cropped = preprocess_image(temp_path)
            if cropped is None:
                result = {
                    'match_found': False,
                    'error': 'No cat detected in the image. Please ensure the image contains a clear view of a cat.',
                    'system_note': 'This system only identifies previously registered cats. New cats must be registered by authorized personnel.'
                }
                last_result = result
                return '''
                <html>
                <head>
                  <title>Processing...</title>
                  <meta http-equiv="refresh" content="2;url=/identify_result">
                  <style>
                    body { background: #fff; font-family: 'Segoe UI', sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
                    .loading-container { display: flex; flex-direction: column; align-items: center; justify-content: center; }
                    .loading-text { color: #f59e0b; font-size: 1.5rem; margin-bottom: 32px; font-weight: 500; }
                    .cat-gif { width: 180px; height: 180px; margin-bottom: 24px; }
                    .spinner { border: 6px solid #f3f3f3; border-top: 6px solid #facc15; border-radius: 50%; width: 48px; height: 48px; animation: spin 1s linear infinite; margin-bottom: 16px; }
                    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                  </style>
                </head>
                <body>
                  <div class="loading-container">
                    <div class="spinner"></div>
                    <div class="loading-text">Analyzing cat image...</div>
                    <img src="/static/catWalking.gif" class="cat-gif" alt="Loading..." />
                  </div>
                </body>
                </html>
                '''
            # Get embedding using our trained contrastive model
            embedding = get_embedding(cropped)
            # Match against database
            result = match_embedding(embedding, db_embeddings, threshold=0.4)
            # Add system information
            result['system_info'] = {
                'auto_registration_enabled': SYSTEM_CONFIG["auto_registration_enabled"],
                'identification_only': SYSTEM_CONFIG["identification_only"],
                'note': 'This system only identifies previously registered cats. New cats must be registered by authorized personnel.'
            }
            # Add model information
            result['model_info'] = {
                'model_type': 'Siamese Network (Contrastive Learning)',
                'base_model': 'EfficientNetB3',
                'accuracy': '69.4%',
                'threshold': 0.4,
                'embedding_dim': 128
            }
            # Add medical information if match found
            matched_id = result.get("matched_id")
            if result.get("match_found") and matched_id in medical_info_db:
                result["medical_info"] = medical_info_db[matched_id]
            # If no match found, provide guidance
            if not result.get("match_found"):
                result["guidance"] = {
                    "message": "No match found in the database.",
                    "next_steps": [
                        "Contact authorized personnel for registration",
                        "Ensure the cat has completed TNR procedures",
                        "Provide clear photos from multiple angles"
                    ],
                    "contact_info": "For registration inquiries, contact the TNR team."
                }
            # Convert numpy types to native Python types for JSON serialization
            for k, v in result.items():
                if isinstance(v, (np.bool_,)):
                    result[k] = bool(v)
                elif isinstance(v, np.ndarray):
                    result[k] = v.tolist()
                elif isinstance(v, np.float32) or isinstance(v, np.float64):
                    result[k] = float(v)
                elif isinstance(v, np.int32) or isinstance(v, np.int64):
                    result[k] = int(v)
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            last_result = result
            return '''
            <html>
            <head>
              <title>Processing...</title>
              <meta http-equiv="refresh" content="2;url=/identify_result">
              <style>
                body { background: #fff; font-family: 'Segoe UI', sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
                .loading-container { display: flex; flex-direction: column; align-items: center; justify-content: center; }
                .loading-text { color: #f59e0b; font-size: 1.5rem; margin-bottom: 32px; font-weight: 500; }
                .cat-gif { width: 180px; height: 180px; margin-bottom: 24px; }
                .spinner { border: 6px solid #f3f3f3; border-top: 6px solid #facc15; border-radius: 50%; width: 48px; height: 48px; animation: spin 1s linear infinite; margin-bottom: 16px; }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
              </style>
            </head>
            <body>
              <div class="loading-container">
                <div class="spinner"></div>
                <div class="loading-text">Analyzing cat image...</div>
                <img src="/static/catWalking.gif" class="cat-gif" alt="Loading..." />
              </div>
            </body>
            </html>
            '''
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            result = {
                'match_found': False, 
                'error': f'Processing error: {str(e)}',
                'system_note': 'This system only identifies previously registered cats.'
            }
            last_result = result
            return '''
            <html>
            <head>
              <title>Processing...</title>
              <meta http-equiv="refresh" content="2;url=/identify_result">
              <style>
                body { background: #fff; font-family: 'Segoe UI', sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
                .loading-container { display: flex; flex-direction: column; align-items: center; justify-content: center; }
                .loading-text { color: #f59e0b; font-size: 1.5rem; margin-bottom: 32px; font-weight: 500; }
                .cat-gif { width: 180px; height: 180px; margin-bottom: 24px; }
                .spinner { border: 6px solid #f3f3f3; border-top: 6px solid #facc15; border-radius: 50%; width: 48px; height: 48px; animation: spin 1s linear infinite; margin-bottom: 16px; }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
              </style>
            </head>
            <body>
              <div class="loading-container">
                <div class="spinner"></div>
                <div class="loading-text">Analyzing cat image...</div>
                <img src="/static/catWalking.gif" class="cat-gif" alt="Loading..." />
              </div>
            </body>
            </html>
            '''
    # (API flow, not browser)
    try:
        # Detect and crop cat from image
        cropped = preprocess_image(temp_path)
        if cropped is None:
            result = {
                'match_found': False,
                'error': 'No cat detected in the image. Please ensure the image contains a clear view of a cat.',
                'system_note': 'This system only identifies previously registered cats. New cats must be registered by authorized personnel.'
            }
            last_result = result
            return render_result_page(result)
        
        # Get embedding using our trained contrastive model
        embedding = get_embedding(cropped)
        
        # Match against database
        result = match_embedding(embedding, db_embeddings, threshold=0.4)
        
        # Add system information
        result['system_info'] = {
            'auto_registration_enabled': SYSTEM_CONFIG["auto_registration_enabled"],
            'identification_only': SYSTEM_CONFIG["identification_only"],
            'note': 'This system only identifies previously registered cats. New cats must be registered by authorized personnel.'
        }
        
        # Add model information
        result['model_info'] = {
            'model_type': 'Siamese Network (Contrastive Learning)',
            'base_model': 'EfficientNetB3',
            'accuracy': '69.4%',
            'threshold': 0.4,
            'embedding_dim': 128
        }
        
        # Add medical information if match found
        matched_id = result.get("matched_id")
        if result.get("match_found") and matched_id in medical_info_db:
            result["medical_info"] = medical_info_db[matched_id]

        # If no match found, provide guidance
        if not result.get("match_found"):
            result["guidance"] = {
                "message": "No match found in the database.",
                "next_steps": [
                    "Contact authorized personnel for registration",
                    "Ensure the cat has completed TNR procedures",
                    "Provide clear photos from multiple angles"
                ],
                "contact_info": "For registration inquiries, contact the TNR team."
            }
        
        # Convert numpy types to native Python types for JSON serialization
        for k, v in result.items():
            if isinstance(v, (np.bool_,)):
                result[k] = bool(v)
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, np.float32) or isinstance(v, np.float64):
                result[k] = float(v)
            elif isinstance(v, np.int32) or isinstance(v, np.int64):
                result[k] = int(v)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        last_result = result
        return render_result_page(result)
        
    except Exception as e:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        result = {
            'match_found': False, 
            'error': f'Processing error: {str(e)}',
            'system_note': 'This system only identifies previously registered cats.'
        }
        last_result = result
        return render_result_page(result)

# Improved results page rendering

def render_result_page(result):
    # Extract values
    match_found = result.get('match_found', False)
    confidence = int(result.get('confidence', 0) * 100)
    matched_id = result.get('matched_id', None)
    error = result.get('error', None)
    guidance = result.get('guidance', {})
    similarity = result.get('similarity', 0)
    model_info = result.get('model_info', {})
    system_info = result.get('system_info', {})
    medical_info = result.get('medical_info', None)
    
    status_color = '#10b981' if match_found else '#ef4444'
    status_text = 'Match Found!' if match_found else 'No Match Found'
    status_icon = '✔️' if match_found else '❌'
    
    # Progress bar for confidence
    progress_html = f'''
      <div style="width: 100%; background: #f3f4f6; border-radius: 8px; height: 24px; margin: 16px 0;">
        <div style="width: {confidence}%; background: {status_color}; height: 100%; border-radius: 8px; transition: width 0.5s;"></div>
      </div>
      <div style="text-align: center; color: {status_color}; font-weight: bold;">Confidence: {confidence}%</div>
    '''
    
    # Medical info HTML
    medical_html = ''
    if medical_info:
        medical_html = '<div style="margin-top: 16px; padding: 12px; background: #fef3c7; border-radius: 8px; color: #b45309; font-size: 1rem;">'
        for k, v in medical_info.items():
            medical_html += f'<div><b>{k.capitalize()}:</b> {v}</div>'
        medical_html += '</div>'
    
    # Guidance HTML
    guidance_html = ''
    if guidance:
        guidance_html = f'''
        <div style="margin-top: 24px; padding: 16px; background: #fef2f2; border-radius: 8px; color: #b91c1c;">
          <b>{guidance.get('message', '')}</b><br/>
          <ul style="margin: 8px 0 0 16px;">
            {''.join(f'<li>{step}</li>' for step in guidance.get('next_steps', []))}
          </ul>
          <div style="margin-top: 8px; font-size: 0.95rem; color: #991b1b;">{guidance.get('contact_info', '')}</div>
        </div>
        '''
    
    # Error HTML
    error_html = ''
    if error:
        error_html = f'<div style="margin-top: 24px; color: #ef4444; font-weight: bold;">{error}</div>'
    
    return f'''
    <html>
    <head>
      <title>Identification Result</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        body {{ background: #fff; font-family: 'Segoe UI', sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }}
        .result-card {{ max-width: 420px; width: 100%; margin: 48px auto; background: #fff; border-radius: 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.10); padding: 40px 32px 32px 32px; display: flex; flex-direction: column; align-items: center; }}
        .status-icon {{ font-size: 3rem; margin-bottom: 12px; color: {status_color}; }}
        .status-text {{ font-size: 2rem; font-weight: bold; color: {status_color}; margin-bottom: 8px; text-align: center; }}
        .cat-id {{ font-size: 1.1rem; color: #888; margin-bottom: 8px; }}
        .model-info {{ font-size: 0.9rem; color: #888; margin-top: 16px; text-align: center; }}
        .back-btn {{ margin-top: 32px; background: #facc15; color: #222; font-weight: bold; border: none; border-radius: 12px; padding: 14px 32px; font-size: 1rem; cursor: pointer; }}
      </style>
    </head>
    <body>
      <div class="result-card">
        <div class="status-icon">{status_icon}</div>
        <div class="status-text">{status_text}</div>
        {progress_html}
        <div class="cat-id">{('Cat ID: ' + matched_id) if matched_id else ''}</div>
        {medical_html}
        {guidance_html}
        {error_html}
        <button class="back-btn" onclick="window.location.href='/'">Back to Home</button>
      </div>
    </body>
    </html>
    '''

@app.route("/identify_result", methods=["GET"])
def identify_result():
    global last_result
    if last_result is None:
        # No result yet, redirect to home
        return '''<html><head><meta http-equiv="refresh" content="0;url=/" /></head><body></body></html>'''
    return render_result_page(last_result)

@app.route("/status", methods=["GET"])
def get_status():
    """Get system status and model information."""
    return jsonify({
        'status': 'operational',
        'system_config': SYSTEM_CONFIG,
        'model_info': {
            'model_type': 'Siamese Network (Contrastive Learning)',
            'base_model': 'EfficientNetB3',
            'accuracy': '69.4%',
            'threshold': 0.4,
            'embedding_dim': 128
        },
        'database_info': {
            'total_embeddings': len(db_embeddings),
            'unique_cats': len(set(e['id'] for e in db_embeddings)) if db_embeddings else 0
        },
        'tensorflow_version': tf.__version__
    })

@app.route("/admin/register", methods=["POST"])
def admin_register_cat():
    """
    Administrative endpoint for registering new cats.
    Requires authorization and should only be used by authorized personnel.
    """
    # In production, implement proper authentication here
    # For now, we'll use a simple API key check
    api_key = request.headers.get('X-API-Key')
    if api_key != 'admin_key_2024':  # In production, use proper authentication
        return jsonify({'error': 'Unauthorized. Admin access required.'}), 401
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    cat_id = request.form.get('cat_id')
    cat_name = request.form.get('cat_name', '')
    notes = request.form.get('notes', '')
    
    if not cat_id:
        return jsonify({'error': 'cat_id is required'}), 400
    
    # Check if cat already exists
    existing_cats = [e['id'] for e in db_embeddings]
    if cat_id in existing_cats:
        return jsonify({'error': f'Cat with ID {cat_id} already exists'}), 409
    
    image = request.files['image']
    temp_path = f"temp_register_{cat_id}.png"
    image.save(temp_path)
    
    try:
        # Detect and crop cat from image
        cropped = preprocess_image(temp_path)
        if cropped is None:
            return jsonify({
                'error': 'No cat detected in the image. Please ensure the image contains a clear view of a cat.'
            }), 400
        
        # Get embedding using our trained contrastive model
        embedding = get_embedding(cropped)
        
        # Add to database
        new_embedding = {
            'id': cat_id,
            'name': cat_name,
            'embedding': embedding,
            'notes': notes,
            'registered_at': datetime.now().isoformat(),
            'registered_by': 'admin'
        }
        
        db_embeddings.append(new_embedding)
        
        # Save updated database
        with open(EMBEDDING_FILE, "wb") as f:
            pickle.dump(db_embeddings, f)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'message': f'Cat {cat_id} registered successfully',
            'cat_info': {
                'id': cat_id,
                'name': cat_name,
                'embedding_dim': len(embedding),
                'registered_at': new_embedding['registered_at']
            }
        })
        
    except Exception as e:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'error': f'Registration failed: {str(e)}'
        }), 500

@app.route("/admin/cats", methods=["GET"])
def admin_list_cats():
    """Administrative endpoint to list all registered cats."""
    # In production, implement proper authentication here
    api_key = request.headers.get('X-API-Key')
    if api_key != 'admin_key_2024':  # In production, use proper authentication
        return jsonify({'error': 'Unauthorized. Admin access required.'}), 401
    
    # Get unique cats with their info
    cats = {}
    for embedding in db_embeddings:
        cat_id = embedding['id']
        if cat_id not in cats:
            cats[cat_id] = {
                'id': cat_id,
                'name': embedding.get('name', ''),
                'notes': embedding.get('notes', ''),
                'registered_at': embedding.get('registered_at', ''),
                'embedding_count': 0
            }
        cats[cat_id]['embedding_count'] += 1
    
    return jsonify({
        'total_cats': len(cats),
        'cats': list(cats.values())
    })

@app.route("/admin/config", methods=["GET"])
def admin_get_config():
    """Get system configuration (admin only)."""
    api_key = request.headers.get('X-API-Key')
    if api_key != 'admin_key_2024':  # In production, use proper authentication
        return jsonify({'error': 'Unauthorized. Admin access required.'}), 401
    
    return jsonify({
        'system_config': SYSTEM_CONFIG,
        'database_stats': {
            'total_embeddings': len(db_embeddings),
            'unique_cats': len(set(e['id'] for e in db_embeddings)) if db_embeddings else 0
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Smart Cat Re-Identification Server on port {port}")
    print(f"Model: Siamese Network (Contrastive Learning) - 69.4% accuracy")
    print(f"Database: {len(db_embeddings)} embeddings loaded")
    app.run(host="0.0.0.0", port=port, debug=True)
