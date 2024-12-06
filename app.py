import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from PIL import Image
import pandas as pd

class_names = ['African_elephant', 'German_shepherd', 'basketball', 'beach_wagon', 'bison', 'bullfrog', 'goldfish', 'jellyfish', 'koala', 'lion', 'pizza', 'salamander', 'snorkel', 'sports_car', 'stopwatch', 'sunglasses', 'tarantula', 'teddy_teddy bear', 'trolleybus', 'volleyball']
params = {'dropout_rate': 0.6, 'l2_rate': 0.4, 'learning_rate': 0.0003, 'epoch': 25}


def EDA():
    data_dir = './data'
    validation_split = 0.2

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=32,
        image_size=(64, 64),
        shuffle=True,
        seed=123,
        validation_split=validation_split,
        subset='training',
        interpolation='bilinear',
    )

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        batch_size=32,
        image_size=(64, 64),
        shuffle=True,
        seed=123,
        validation_split=validation_split,
        subset='validation',
        interpolation='bilinear',
    )

    # Preprocess inputs
    train_data = train_data.map(lambda x, y: (preprocess_input(x), y))
    test_data = test_data.map(lambda x, y: (preprocess_input(x), y))
    
    return train_data, test_data


def build_model(params, train_data, test_data):
    dropout_rate = params['dropout_rate']
    l2_rate = params['l2_rate']
    learning_rate = params['learning_rate']
    epoch = params['epoch']

    try:
        # Load ResNet50 as base model
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

        # Freeze the first 100 layers of the ResNet50 model
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False

        # Build the CNN model
        CNN_model = keras.Sequential([
            base_model,
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(l2_rate)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(l2_rate)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(20, activation='softmax')
        ])

        CNN_model.summary()

        # Compile the model
        adam = Adam(learning_rate=learning_rate)
        CNN_model.compile(loss="sparse_categorical_crossentropy",
                          optimizer=adam,
                          metrics=["accuracy"])

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,
            patience=6,
            min_lr=1e-5,
            verbose=1
        )

        # Train the model
        history_CNN_model = CNN_model.fit(
            train_data,
            validation_data=test_data,
            epochs=epoch,
            callbacks=[lr_scheduler, early_stopping],
            verbose=1
        )

        # Evaluate the model
        CNN_model.evaluate(test_data)

        return history_CNN_model, CNN_model

    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return None


@st.cache(allow_output_mutation=True)
def train_and_cache_model(params, train_data, test_data):
    # Train the model and return both the history and the model
    history, CNN_model = build_model(params, train_data, test_data)
    return history, CNN_model


def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((64, 64))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def display_history(history):
    st.subheader('Model History')
    accuracy_data = pd.DataFrame({
        'Epochs': np.arange(1, len(history.history['accuracy']) + 1),
        'Training Accuracy': history.history['accuracy'],
        'Validation Accuracy': history.history['val_accuracy']
    })
    
    st.subheader('Training vs Validation Accuracy')
    st.line_chart(accuracy_data.set_index('Epochs'))

    loss_data = pd.DataFrame({
        'Epochs': np.arange(1, len(history.history['loss']) + 1),
        'Training Loss': history.history['loss'],
        'Validation Loss': history.history['val_loss']
    })

    st.subheader('Training vs Validation Loss')
    st.line_chart(loss_data.set_index('Epochs'))


def streamlit_app():
    st.title("CNN Image Classifier")
    st.markdown("Upload an image to classify:")

    # Display training history if it exists
    if 'history' in st.session_state:
        history = st.session_state.history  # Retrieve history from session_state
        display_history(history)  # Display the plots

    # Train the model if not done already
    if 'CNN_model' not in st.session_state:
        with st.spinner('Training CNN Model...'):
            train_data, test_data = EDA()  # Load data
            history, CNN_model = train_and_cache_model(params, train_data, test_data)  # Cache the model and history
            # Save model and history in session_state
            st.session_state.history = history
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
