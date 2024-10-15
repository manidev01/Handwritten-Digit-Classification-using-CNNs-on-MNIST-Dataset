import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('cnn_mnist_model.h5')

# Set up the Streamlit app title and instructions
st.title('Digit Classification with CNN')
st.write('Upload a handwritten digit image (28x28), and the model will predict the digit!')

# File uploader to allow users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Preprocess the uploaded image
    img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)

    # Display the uploaded image
    st.image(img.reshape(28, 28), caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Show the predicted class
    st.write(f'Predicted class: {predicted_class}')
