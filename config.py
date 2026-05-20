# Configuration Settings for Sign Language Detection System

import os
import sqlite3

# Dynamically load WORDS from local database
DB_NAME = "sign_database.db"
WORDS_LIST = []
try:
    if os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT word FROM gestures")
        WORDS_LIST = sorted([row[0].lower() for row in cursor.fetchall()])
        conn.close()
except:
    pass

if not WORDS_LIST:
    WORDS_LIST = ["hello", "thanks", "yes", "no", "help"]

WORDS = WORDS_LIST
NUM_CLASSES = len(WORDS)

# Sequence configuration (Number of frames needed for an LSTM prediction)
SEQUENCE_LENGTH = 30 

# MediaPipe hands produces 21 landmarks per hand, each with (x, y) coordinates
# We use 42 numerical features per frame
LANDMARKS_COUNT = 42 

# Path to save and load the trained Keras model
MODEL_PATH = "sign_model.keras"

# HuggingFace dataset identifier
DATASET_NAME = "akasheroor/American-Sign-Language-Dataset"
