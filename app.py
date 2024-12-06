import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import os 

class_names = ['African_elephant', 'German_shepherd', 'basketball', 'beach_wagon', 'bison', 'bullfrog', 'goldfish', 'jellyfish', 'koala', 'lion', 'pizza', 'salamander', 'snorkel', 'sports_car', 'stopwatch', 'sunglasses', 'tarantula', 'teddy_teddy bear', 'trolleybus', 'volleyball']
params = {'dropout_rate': 0.6, 'l2_rate': 0.4, 'learning_rate': 0.0003, 'epoch': 25}




def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((64, 64))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def streamlit_app():
    st.markdown("Upload an image to classify:")

    if 'CNN_model' not in st.session_state:
        with st.spinner('Training CNN Model...'):
            CNN_model = load_model('CNN_model.h5')
            st.session_state.CNN_model = CNN_model



    # Upload image for prediction
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # Preprocess and predict
        img_array = preprocess_image(uploaded_image)
        predictions = st.session_state.CNN_model.predict(img_array)

        # Get the class with the highest probability
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Display prediction
        st.subheader('Prediction')
        st.write(class_names[predicted_class])

        st.subheader('Prediction Probabilities')
        st.write(predictions)


streamlit_app()
