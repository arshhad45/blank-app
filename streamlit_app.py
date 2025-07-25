# app.py

import streamlit as st
import numpy as np
from PIL import Image
import os

# Handle gdown installation and import
try:
    import gdown
except ImportError:
    os.system('pip install gdown')
    import gdown

import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# --- Configuration ---
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")

# Google Drive model file ID (replace this with your actual file ID)
MODEL_ID = "16SJDR7VjyEaFkIDX8ys1eAjK1jCm1m4X"  # <-- REPLACE with your actual file ID
MODEL_URL = f"https://drive.google.com/file/d/16SJDR7VjyEaFkIDX8ys1eAjK1jCm1m4X/view?usp=sharing={MODEL_ID}"
MODEL_PATH = "custom_cnn_model.h5"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = load_model(MODEL_PATH)
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# --- UI ---
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI brain scan to predict the tumor type.")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

# Image Preprocessing
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict and display results
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("⏳ Classifying...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🔍 Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
