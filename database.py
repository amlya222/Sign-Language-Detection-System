import sqlite3
import os
import uuid
import numpy as np
import cv2
import mediapipe as mp
import datetime

DB_NAME = "sign_database.db"
RAW_DATA_DIR = "dataset/raw"
KEYPOINTS_DIR = "dataset/keypoints"

def init_db():
    """Initializes the SQLite database and creates the necessary directories."""
    # Ensure directories exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(KEYPOINTS_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create gestures table to store metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gestures (
            id TEXT PRIMARY KEY,
            word TEXT NOT NULL,
            contributor TEXT,
            video_path TEXT NOT NULL,
            keypoints_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending'
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"[INFO] Database initialized at {DB_NAME}")

def save_gesture_video(word, video_bytes, contributor="Anonymous"):
    """
    Saves a raw video file to the file system and logs metadata to SQLite.
    Returns the generated unique gesture ID.
    """
    word = word.upper().strip()
    gesture_id = str(uuid.uuid4())
    
    # Ensure word directory exists in raw data
    word_dir = os.path.join(RAW_DATA_DIR, word)
    os.makedirs(word_dir, exist_ok=True)
    
    video_filename = f"{gesture_id}.webm"
    video_path = os.path.join(word_dir, video_filename)
    
    # Write the video bytes to disk
    with open(video_path, 'wb') as f:
        f.write(video_bytes)
        
    # Log metadata to SQLite
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO gestures (id, word, contributor, video_path) VALUES (?, ?, ?, ?)",
        (gesture_id, word, contributor, video_path)
    )
    conn.commit()
    conn.close()
    
    print(f"[INFO] Saved raw video for '{word}' to {video_path}")
    return gesture_id, video_path

def extract_and_save_keypoints(gesture_id, video_path, word):
    """
    Processes a saved video to extract MediaPipe keypoints and saves them 
    as a numpy .npy file for ML training. Updates the SQLite database.
    """
    word = word.upper().strip()
    
    # Ensure word directory exists in keypoints data
    word_keypoints_dir = os.path.join(KEYPOINTS_DIR, word)
    os.makedirs(word_keypoints_dir, exist_ok=True)
    
    keypoints_filename = f"{gesture_id}.npy"
    keypoints_path = os.path.join(word_keypoints_dir, keypoints_filename)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    frames_keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hands
        results = hands.process(image_rgb)
        
        # We need 42 features per frame (21 landmarks x 2 coordinates)
        frame_landmarks = np.zeros(42)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks_list = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            frame_landmarks = np.array(landmarks_list).flatten()
            
        frames_keypoints.append(frame_landmarks)
        
    cap.release()
    hands.close()
    
    # Convert sequence to numpy array and save
    frames_keypoints = np.array(frames_keypoints)
    np.save(keypoints_path, frames_keypoints)
    
    # Update SQLite database with the keypoints path
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE gestures SET keypoints_path = ?, status = 'processed' WHERE id = ?",
        (keypoints_path, gesture_id)
    )
    conn.commit()
    conn.close()
    
    print(f"[INFO] Processed keypoints for '{word}' to {keypoints_path}")
    return keypoints_path

if __name__ == "__main__":
    init_db()
