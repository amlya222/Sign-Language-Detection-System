import mediapipe as mp
import cv2
import numpy as np
from datasets import load_dataset
from config import WORDS, SEQUENCE_LENGTH, DATASET_NAME
import tempfile
import os

# --- CRITICAL WINDOWS PATCH ---
# We bypass the datasets library entirely because it forces torchcodec
# and does not map labels correctly for this specific ASL dataset repository.
import pandas as pd
from huggingface_hub import hf_hub_download
import shutil
# ------------------------------

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def extract_landmarks(frame):
    """
    Takes an OpenCV frame (BGR form) or Numpy array, processes it via MediaPipe,
    and returns a flattened array of 21 (x, y) coordinates for the first detected hand.
    Returns array of 42 zeros if no hand detected.
    """
    if hasattr(frame, 'convert'):
        frame = np.array(frame)
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
        return landmarks
        
    return np.zeros(42)

def stream_and_preprocess_dataset(max_samples_per_word=100):
    """
    Downloads exactly what we need directly from HuggingFace to avoid downloading 100GB.
    Filters only common words and extracts landmarks.
    """
    print(f"Connecting to HuggingFace repository {DATASET_NAME} metadata...")
    
    try:
        # Load the index CSV directly from huggingface
        csv_url = f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/dataset.csv"
        df = pd.read_csv(csv_url)
    except Exception as e:
        print(f"Error loading dataset.csv: {e}")
        return np.array([]), np.array([])
        
    X_data = []
    Y_data = []
    
    print("Extracting targeted features off remote servers...")
    
    for word_idx, target_word in enumerate(WORDS):
        # Find all videos for this word
        word_videos = df[df['word'].str.lower() == target_word]
        
        # Take up to max_samples
        sampled_videos = word_videos.head(max_samples_per_word)
        
        count = 0
        for _, row in sampled_videos.iterrows():
            video_rel_path = row['video_path']
            # e.g., 'part_5/train.mp4'
            
            # Download that single video file
            try:
                local_vid_path = hf_hub_download(repo_id=DATASET_NAME, repo_type="dataset", filename=video_rel_path)
            except Exception as e:
                print(f"Skipping {target_word} video due to download error: {e}")
                continue
                
            sequence = []
            cap = cv2.VideoCapture(local_vid_path)
            while cap.isOpened() and len(sequence) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret:
                    break
                landmarks = extract_landmarks(frame)
                sequence.append(landmarks)
            cap.release()
            
            # Discard extremely short/empty sequences
            if len(sequence) < 5:
                continue
                
            # Pad sequence if the video was shorter than required frames
            while len(sequence) < SEQUENCE_LENGTH:
                sequence.append(np.zeros(42))
                
            X_data.append(sequence)
            Y_data.append(word_idx)
            count += 1
            
        print(f"Successfully processed targeted word '{target_word}' ({count}/{max_samples_per_word})")
            
    return np.array(X_data), np.array(Y_data)

def load_local_dataset():
    """
    Loads custom recorded `.npy` features directly from the local SQLite dataset storage.
    Automatically generates overlapping sequences to maximize training data footprint.
    """
    import sqlite3
    from config import DB_NAME, WORDS
    
    print(f"Connecting to Local Database ({DB_NAME})...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT word, keypoints_path FROM gestures WHERE status='processed'")
    records = cursor.fetchall()
    conn.close()
    
    X_data = []
    Y_data = []
    
    for word, kp_path in records:
        word_lower = word.lower()
        if word_lower not in WORDS:
            continue
            
        word_idx = WORDS.index(word_lower)
        
        try:
            frames_keypoints = np.load(kp_path)  # shape: (num_frames, 42)
        except Exception as e:
            print(f"Skipping {kp_path}: {e}")
            continue
            
        num_frames = len(frames_keypoints)
        if num_frames < 5:
            continue
            
        stride = 3 # Overlapping sequences every 3 frames
        for start_idx in range(0, max(1, num_frames - SEQUENCE_LENGTH + 1), stride):
            end_idx = start_idx + SEQUENCE_LENGTH
            sequence = list(frames_keypoints[start_idx:end_idx])
            
            # Pad sequence if the video excerpt was short
            while len(sequence) < SEQUENCE_LENGTH:
                sequence.append(np.zeros(42))
                
            X_data.append(sequence)
            Y_data.append(word_idx)
            
    return np.array(X_data), np.array(Y_data)
