
# Real-Time Sign Language Detection System

An AI-powered application designed for real-time translation of American Sign Language (ASL) into English text and spoken voice. Built using Python, TensorFlow, MediaPipe, and OpenCV.

## Features
- **Real-Time Translation**: Captures webcam feeds and uses MediaPipe hand landmarks to classify gestures instantly.
- **Deep Learning Model**: Uses a Sequential LSTM architecture trained on HuggingFace datasets.
- **Accessible UI**: A clean Tkinter interface showing camera feed, target word, confidence scores, and sentence history.
- **Text-to-Speech (TTS)**: Automatically speaks the predicted word aloud.
- **Log generation**: Saves predictions to `predictions.log` for future review.

## System Requirements
- Python 3.8 to 3.11 recommended.
- A functional webcam.
- Active Internet connection (for initial dataset streaming during training).

## Installation

### 1. Install Dependencies
Run the following command to download all required machine learning and UI libraries:
```bash
pip install -r requirements.txt
```

*(Note: Depending on your OS, you might need to run `pip3` instead of `pip`)*

## Usage

### 2. Train the Model
Before running the Real-Time detection, you must stream the dataset from HuggingFace and train the model. The dataset will be partially loaded in chunks to preserve your local disk space.
```bash
python main.py --train
```
*(This will compute and securely save `sign_model.keras` to your folder when finished)*

### 3. Launch the Translator
Once your model file `sign_model.keras` has been generated, you can launch the GUI to begin translating.
```bash
python main.py
```

## Modifying Vocabulary
You can alter which words the AI learns to recognize by opening `config.py` and modifying the `WORDS` array. 

---
*Created as a final-year engineering project demonstration.*

# Sign-Language-Detection-System

