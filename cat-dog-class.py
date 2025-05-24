# -- Imports --
import streamlit as st
import numpy as np
import os
import urllib.request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -- Page Configuration --
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

# -- Constants --
MODEL_URL = "https://huggingface.co/datasets/dnforimaco/cat-dog-classification/resolve/main/cat-dog_model.h5"
MODEL_PATH = "cat-dog_model.h5"
IMG_SIZE = (150, 150)  # Change if your model uses a different input size
CLASS_NAMES = ['Cat', 'Dog']  # Assumes binary classification: 0 = Cat, 1 = Dog

# -- Load Model with Caching --
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = load_model(MODEL_PATH)
    return model

model = download_and_load_model()

# -- UI Styling --
st.markdown("""
    <style>
        .stButton>button {
            background-color: #007bff;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            color: #888;
            font-size: 0.85rem;
        }
    </style>
""", unsafe_allow_html=True)

# -- App Title and Upload --
st.title("🐱🐶 Cat vs Dog Classifier")
st.markdown("Upload an image to determine whether it's a **cat** or a **dog**.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=False, width=300)

    # Preprocess
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("🔍 Classifying..."):
        prediction = model.predict(img_array)[0]
        predicted_class = CLASS_NAMES[int(round(prediction[0]))]  # Assuming output is a sigmoid score
        confidence = prediction[0] * 100 if predicted_class == 'Dog' else (1 - prediction[0]) * 100

    # Display Results
    st.success(f"🎯 **Prediction:** `{predicted_class}`")
    st.metric("🔒 Confidence", f"{confidence:.2f} %")
