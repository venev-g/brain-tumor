import numpy as np
import tensorflow as tf
import streamlit as st
import os
from PIL import Image
import io

# Load the trained model
cnn = tf.keras.models.load_model('braintumor.keras')
# Create the Streamlit app
st.title("Brain Tumor Detection")
st.write("Upload an image to predict if there is a brain tumor.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load image data from file
    img = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    img = Image.open(uploaded_file).convert('RGB')

    # Preprocess the image
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    result = cnn.predict(img_array)
    prediction = 'Tumor Present' if result[0][0] > 0.5 else 'No Tumor'
    
    st.write(f"Prediction: {prediction}")
