import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

st.title("Hand Gesture Recognition")
st.write("Upload an image of a hand gesture to classify it.")

# Load the trained model
model = load_model('hand_gesture_model.h5')
print("Model loaded successfully.")

# Define class names (digits 0-9)
class_names = [str(i) for i in range(10)]

# Add file uploader component
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process the uploaded file
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
     # Read the image using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE) # Load as grayscale

    # Preprocess the image for the model
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1) # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Display a message indicating the image is ready for prediction
    st.write("Image loaded and preprocessed. Ready for prediction.")

    # Make a prediction when the user uploads an image
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]

    # Display the prediction result
    st.subheader("Prediction:")
    st.write(f"The hand gesture is likely: **{predicted_class_name}**")

    # Optional: Display the confidence scores for all classes
    # st.write("Confidence Scores:")
    # for i, prob in enumerate(prediction[0]):
    #     st.write(f"  {class_names[i]}: {prob:.4f}")