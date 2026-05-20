import sqlite3
import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, send_from_directory
from database import save_gesture_video, extract_and_save_keypoints, init_db, DB_NAME, RAW_DATA_DIR, KEYPOINTS_DIR
from detection import SignDetector
from dotenv import load_dotenv

# Load environmental variables
load_dotenv()

# Initialize database on startup
init_db()

# Global detector instance to keep model in memory
print("[INFO] Loading Sign Language Model...")
global_detector = SignDetector()

# Serve static files from the current directory
app = Flask(__name__, static_url_path='', static_folder='.')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/detect', methods=['POST'])
def run_detection():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
        
    try:
        image_b64 = data.get('image')
        # format: "data:image/jpeg;base64,/9j/4AA..."
        encoded_data = image_b64.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect the signs with global detector
        _, word, conf = global_detector.detect_signs_real_time(frame)
        
        return jsonify({"success": True, "word": word, "confidence": conf})
    except Exception as e:
        print(f"[Detect Error] {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/add_gesture', methods=['POST'])
def add_gesture():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    word = request.form.get('word', '').strip()
    
    if not word:
        return jsonify({"error": "Word or phrase is required"}), 400
        
    print(f"[API] Received gesture contribution for: {word}")
    
    # Read the bytes of the uploaded video
    video_bytes = video_file.read()
    
    try:
        # Save raw video to SQLite and File System
        gesture_id, video_path = save_gesture_video(word, video_bytes)
        
        # Extract features and save numpy array
        keypoints_path = extract_and_save_keypoints(gesture_id, video_path, word)
        
        return jsonify({
            "success": True, 
            "message": f"Successfully saved and processed '{word}'!",
            "gesture_id": gesture_id
        })
    except Exception as e:
        print(f"[API Error] {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/available_gestures', methods=['GET'])
def get_available_gestures():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT word FROM gestures")
        words = [row[0] for row in cursor.fetchall()]
        conn.close()
        return jsonify({"success": True, "words": words})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/library', methods=['GET'])
def get_library():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT word, COUNT(*) as count FROM gestures GROUP BY word ORDER BY created_at DESC")
        results = [{"word": row[0], "count": row[1]} for row in cursor.fetchall()]
        conn.close()
        return jsonify({"success": True, "library": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/delete_word', methods=['POST'])
def delete_word():
    data = request.json
    word = data.get('word', '').upper().strip()
    if not word:
        return jsonify({"error": "No word provided"}), 400
        
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM gestures WHERE word = ?", (word,))
        conn.commit()
        conn.close()
        
        # Remove physical files
        import shutil
        raw_dir = os.path.join(RAW_DATA_DIR, word)
        kp_dir = os.path.join(KEYPOINTS_DIR, word)
        if os.path.exists(raw_dir):
            shutil.rmtree(raw_dir)
        if os.path.exists(kp_dir):
            shutil.rmtree(kp_dir)
            
        print(f"[API] Deleted all records for '{word}'")
        return jsonify({"success": True, "message": f"Deleted {word}"})
    except Exception as e:
        print(f"[API Error] {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate_sentence', methods=['POST'])
def generate_sentence():
    data = request.json
    api_key = data.get('api_key', '').strip()
    words = data.get('words', [])
    
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        
    if not api_key:
        return jsonify({"error": "Please provide a valid Gemini API Key or add it to .env."}), 400
    if not words:
        return jsonify({"error": "No words detected yet."}), 400
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Dynamically find a valid model for this API key and Region
        valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if not valid_models:
            return jsonify({"error": "No valid Gemini generation models found for this API key."}), 400
            
        # Prefer a fast 'flash' model, otherwise just pick the first available one
        model_name = valid_models[0]
        for name in valid_models:
            if 'flash' in name:
                model_name = name
                break
                
        model = genai.GenerativeModel(model_name)
        
        prompt = (
            "You are a sign language translator. "
            "Convert the following sequence of sign language words into a proper, "
            "grammatically correct, and natural English sentence. "
            "Only reply with the sentence, nothing else.\n\n"
            f"Words: {' '.join(words)}"
        )
        
        response = model.generate_content(prompt)
        return jsonify({"success": True, "sentence": response.text.strip()})
    except Exception as e:
        print(f"[AI Error] {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("[INFO] Starting Backend Server on http://localhost:5000")
    app.run(port=5000, debug=True)
    # Restart trigger comment
