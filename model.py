import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from config import SEQUENCE_LENGTH, NUM_CLASSES, MODEL_PATH, LANDMARKS_COUNT
from dataset import load_local_dataset

def build_lstm_model(input_shape, num_classes):
    """
    Builds a sequential LSTM model using Tensorflow/Keras
    Input shape is (sequences, landmarks)
    """
    model = Sequential()
    
    # Input sequence layers
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    
    # Dense classification layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    """
    Loads local dataset created by the user from SQLite,
    splits into train/test and trains deep learning model
    """
    print("Initiating local dataset preprocessing...")
    # Gather sequences directly from the dataset/keypoints folder
    X, Y = load_local_dataset()
    
    if len(X) == 0:
        print("Error: Could not extract features from local dataset. Please record more gestures using the Add Sign page.")
        return
        
    print(f"Data gathered successfully. Shape X: {X.shape}, Y: {Y.shape}")
    
    # Train test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print("Building model architecture...")
    model = build_lstm_model((SEQUENCE_LENGTH, LANDMARKS_COUNT), NUM_CLASSES)
    model.summary()
    
    # Callbacks to save the best version of our .h5 / .keras file
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    
    print("Start training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=4,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint]
    )
    
    print(f"Training accurate. Final Training Accuracy: {history.history['accuracy'][-1]:.2f}")
    print(f"Model saved to file: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
