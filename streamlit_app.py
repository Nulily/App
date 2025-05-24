import streamlit as st
import numpy as np
import os
import urllib.request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(page_title="Mango Leaf Disease Classifier", layout="centered")

# -- Configuration --
MODEL_URL = "https://drive.google.com/uc?export=download&id=1Re5Xxbp_PF6U6tBrBpqFnzo3I3ha5gyI"
MODEL_PATH = "mangoleafdisease_classification_mobilenetv2_95pct.h5"
IMG_SIZE = (224, 224)  # Input size for MobileNetV2
CLASS_NAMES = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Powdery Mildew', 'Sooty Mould', 'Healthy' ]  # Update if you used different labels

# -- Model loader with caching --
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = load_model(MODEL_PATH)
    return model

model = download_and_load_model()

# -- Streamlit UI --
st.title("Mango Leaf Disease")
st.write("Upload an image to identify the Leaf Disease.")

uploaded_file = st.file_uploader("Upload image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    with st.spinner("Classifying..."):
        prediction = model.predict(img_array)[0]
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.success(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")
