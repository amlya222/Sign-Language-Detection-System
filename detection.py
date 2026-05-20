import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from config import SEQUENCE_LENGTH, WORDS, MODEL_PATH, LANDMARKS_COUNT
import os
import pyttsx3

class SignDetector:
    def __init__(self):
        # Hot-reload config words because the SQLite DB might have changed
        import config
        import importlib
        importlib.reload(config)
        self.words = config.WORDS
        
        # Open MediaPipe setup
        self.mp_hands = mp.solutions.hands
        # We only need one hand for basic word tracking to keep the model fast
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Sequence data (buffer of the last 30 frames)
        self.sequence = []
        
        # Debounce and Stabilization trackers
        self.frame_counter = 0
        self.recent_predictions = []
        self.last_prediction_text = "Waiting..."
        self.last_confidence_val = 0.0
        
        # Load the saved model (tf/keras)
        self.model = None
        self.is_model_loaded = False
        self.load_model()
        
        # Text To Speech setup
        self.engine = pyttsx3.init()
        # Optional: Slow down voice slightly for clarity
        rate = self.engine.getProperty('rate')
        self.engine.setProperty('rate', rate - 30)

    def load_model(self):
        """
        Gently loads the trained .h5 or .keras model from disk
        """
        if os.path.exists(MODEL_PATH):
            try:
                self.model = tf.keras.models.load_model(MODEL_PATH)
                self.is_model_loaded = True
                print(f"Loaded sign language model from: {MODEL_PATH}")
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print("Model not loaded because it hasn't been trained yet.")

    def text_to_speech(self, text):
        """
        Read predicted word aloud using pyttsx3.
        Runs in background thread so it doesn't interrupt OpenCV.
        """
        self.engine.say(text)
        self.engine.runAndWait()

    def detect_signs_real_time(self, frame):
        """
        Given a video frame, uses MediaPipe for feature extraction and TensorFlow for prediction.
        Returns the annotated frame, the detected word, and confidence %.
        """
        prediction_text = "Waiting..."
        confidence_percent = 0.0
        
        # Guard clause - If model missing, abort prediction
        if not self.is_model_loaded:
            return frame, "Train Model First", 0.0

        # Processing MediaPipe hand tracking natively on frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)
        
        # Array to store current frame landmarks
        landmarks_array = np.zeros(LANDMARKS_COUNT)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            # Draw skeletons for debug/UI feedback
            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Extract features for current frame
            landmarks_list = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            landmarks_array = np.array(landmarks_list).flatten()
            
            # Buffer the frame features
            self.sequence.append(landmarks_array)
            # Clip buffer to the desired SEQUENCE_LENGTH (e.g. 30 frames)
            self.sequence = self.sequence[-SEQUENCE_LENGTH:]
            self.frame_counter += 1
            
            # Send prediction sequence to our LSTM model if buffer full
            # Only run inference every 3 frames to drastically reduce lag and CPU burn
            if len(self.sequence) == SEQUENCE_LENGTH and self.frame_counter % 3 == 0:
                # Expand dims to mimic batch shape: (1, sequences, feature_count)
                model_input = np.expand_dims(self.sequence, axis=0)
                predictions = self.model.predict(model_input, verbose=0)[0]
                
                # Get index of max probability
                pred_idx = np.argmax(predictions)
                confidence_val = predictions[pred_idx]
                
                # We require a much higher threshold (85%) to avoid random garbage inferences
                if confidence_val > 0.85:
                    word = self.words[pred_idx].upper() if pred_idx < len(self.words) else "UNKNOWN"
                    self.recent_predictions.append(word)
                    
                    # Keep a trailing history of the last 5 successful high-confidence inferences
                    if len(self.recent_predictions) > 5:
                        self.recent_predictions.pop(0)
                        
                    # Smoothe the UI by requiring a strict majority vote (4 out of 5 frames must agree)
                    if self.recent_predictions.count(word) >= 4:
                        self.last_prediction_text = word
                        self.last_confidence_val = float(confidence_val * 100)
                    else:
                        self.last_prediction_text = f"UNCLEAR ({word})"
                        self.last_confidence_val = float(confidence_val * 100)
                else:
                    # Below 85% confidence is treated as a transition/unclear
                    idx_word = self.words[pred_idx].upper() if pred_idx < len(self.words) else "UNKNOWN"
                    self.last_prediction_text = f"UNCLEAR ({idx_word})"
                    self.last_confidence_val = float(confidence_val * 100)
                    # When confidence drops, optionally wipe history so it doesn't bleed into next gesture
                    self.recent_predictions = []
                    
            print(f"[Real-time] {self.last_prediction_text} | {self.last_confidence_val:.1f}% | Sequence: {len(self.sequence)}/{SEQUENCE_LENGTH}")
            
            prediction_text = self.last_prediction_text
            confidence_percent = self.last_confidence_val
        else:
            # If the user put their hand down, wipe the memory buffer
            # This prevents stitching two separate gestures together and getting stuck!
            self.sequence = []
            self.recent_predictions = []
            prediction_text = "Waiting..."
            confidence_percent = 0.0
                    
        return frame, prediction_text, confidence_percent
