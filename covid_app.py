import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Create the title for the App
st.title('COVID-19 scan classification')
st.write(f'Upload an Image of scan and we will predict if the person has COVID-19, Pneumonia or None')

# create a file Uploader
uploaded_file = st.file_uploader('Upload an Image...', type=['jpg', 'png', 'jpeg'])

# check a file uploader
if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='uploaded Image')
    st.write("")

    # Preprocess the image
    img = np.array(image)

    # Change image to RGB    
    img = image.convert('RGB')

    # resize the image
    img = tf.image.resize(img, (128,128))

    # normalize the image
    img = img/255.0
    img = np.expand_dims(img, axis = 0)
    st.write(f'{img.shape}')

    # Load the trained model
    model = load_model('image_model01.h5')


    # Make predictions using your model
    prediction = model.predict(img)

    # Get the class index with the highest predicted probability
    predicted_class_index = np.argmax(prediction)

    # Define your class labels
    class_labels = ['Covid', 'Normal', 'Viral Pneumonia']

    # Assign the label based on the predicted class index
    label = class_labels[predicted_class_index]

    # Display the prediction
    st.write(f'## The predicted Image: {label}')


