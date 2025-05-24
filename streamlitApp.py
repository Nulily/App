# -- Imports --
import streamlit as st
import numpy as np
import os
import urllib.request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(page_title="Mango Leaf Disease Classifier", layout="centered")

# -- Configuration --
MODEL_URL = "https://huggingface.co/datasets/Nulily/mangoleafdisease_classification_mobilenetv2_95pct/resolve/main/mangoleafdisease_classification_mobilenetv2_95pct.h5"
MODEL_PATH = "mangoleafdisease_classification_mobilenetv2_95pct.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
    'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
]

# -- Load Model with Caching --
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = load_model(MODEL_PATH)
    return model

model = download_and_load_model()

# -- UI Design -- 
st.markdown("""
    <style>
        .stButton>button {
            background-color: #28a745;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #218838;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            color: #888;
            font-size: 0.85rem;
        }
    </style>
""", unsafe_allow_html=True)

# -- Streamlit App UI --
st.title("Mango Leaf Disease Identifier 🍃")
st.markdown("Upload an image of a mango leaf to identify the disease.")

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=False, width=300)

    # Preprocess the image
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🔍 Identifying Leaf State..."):
        prediction = model.predict(img_array)[0]
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.success(f"🎯 **Prediction:** `{predicted_class}`")
    st.metric("🔒 Confidence", f"{confidence:.2f} %")
