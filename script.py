import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import uvicorn
import threading
import time
import requests
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.preprocessing import image
from PIL import Image
from io import BytesIO

# Dataset Paths
DATASET_PATH = "C:/Users/GIgabyte/Facial_Gender_Detection"
TRAIN_DIR = os.path.join("C:/Users/GIgabyte/Facial_Gender_Detection/DATASET1/Train")
VALID_DIR = os.path.join("C:/Users/GIgabyte/Facial_Gender_Detection/DATASET1/validation")
TEST_DIR = os.path.join("C:/Users/GIgabyte/Facial_Gender_Detection/DATASET1/TEST")
MODEL_PATH = "gender_classification_model.h5"

# Initialize FastAPI
app = FastAPI()

def train_model():
    """Train the model if the file does not exist."""
    print("\n🚀 Training new model...")
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=25, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')
    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, batch_size=64, class_mode='binary', target_size=(64,64))
    validation_generator = validation_datagen.flow_from_directory(VALID_DIR, batch_size=64, class_mode='binary', target_size=(64,64))
    
    # Build Model
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(64,64,3)),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(train_generator, validation_data=validation_generator, epochs=10)
    model.save(MODEL_PATH)
    print("\n✅ Model trained and saved successfully!")
    return model

# Load or Train Model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("\n✅ Model loaded successfully!")
else:
    model = train_model()

labels = {0: "Female", 1: "Male"}

@app.post("/predict/")
async def predict_gender(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = np.array(Image.open(BytesIO(contents)))
        img = cv2.resize(img, (64,64)) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)[0][0]
        predicted_label = labels[int(round(prediction))]
        return {"prediction": predicted_label}
    except Exception as e:
        return {"error": str(e)}

def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

thread = threading.Thread(target=run_fastapi, daemon=True)
thread.start()
time.sleep(3)

# Streamlit Frontend
st.title("🎥 Real-Time Gender Classification")
st.write("Capture an image using your webcam and classify gender.")

image_file = st.camera_input("📷 Capture Image")

if image_file:
    img = Image.open(image_file)
    img = np.array(img)
    _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img_bytes = img_encoded.tobytes()

    with st.spinner("Predicting..."):
        try:
            response = requests.post("http://127.0.0.1:8000/predict/", files={"file": img_bytes})
            if response.status_code == 200:
                result = response.json()
                st.success(f"🟢 Predicted Gender: **{result['prediction']}**")
            else:
                st.error("❌ Error: Could not classify image. Try again.")
        except requests.exceptions.ConnectionError:
            st.error("🚨 FastAPI server is not running! Restart the script.")
