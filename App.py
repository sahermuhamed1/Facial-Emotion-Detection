import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
import tempfile

# Load the model and compile it
model = load_model('emotion_detection_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

# Streamlit app setup
st.title('Facial Emotion Detection App 🎭')
st.write('This is a facial emotion detection app built with a Convolutional Neural Network (CNN) and deep learning.')

# Function to preprocess the image
def preprocess_image(image_path):
    """Preprocesses the image for prediction."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=(0, -1)) # Shape: (1, 48, 48, 1)
    return image

# Function to predict the class of the image
def predict_image(image_path):
    """Predicts the label for the uploaded image using the trained model."""
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    return prediction

# Allow user to upload image
option = st.radio('Choose how to upload image:', ('Upload Image', 'Open Camera'))

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Display the uploaded image and make it smaller
        image = Image.open(uploaded_file)
        # make the image display smaller
        st.image(image, caption='Uploaded Image', width=300)  # Set width to 300 pixels (or any desired value)
        
        
        # Make prediction
        prediction = predict_image(temp_file_path)
        predicted_class = np.argmax(prediction)
        
        # Display the prediction
        st.write(f"Prediction: {le.inverse_transform([predicted_class])[0]}")
    else:
        st.write('Please upload an appropriate image to predict.')


else: # Capture image from camera using Streamlit's built-in camera input
    captured_image = st.camera_input("Take a picture")
    if captured_image is not None:
        # Save the captured image temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(captured_image.read())
            temp_file_path = temp_file.name
        
        # Display the captured image
        st.image(captured_image, caption='Captured Image', width=300)
        
        # Make prediction
        prediction = predict_image(temp_file_path)
        predicted_class = np.argmax(prediction)
        
        # Display the prediction
        st.write(f"Prediction: {le.inverse_transform([predicted_class])[0]}")
    else:
        st.write("Please take a picture to predict.")