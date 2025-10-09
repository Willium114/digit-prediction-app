import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("mnist_ann_model.keras")

# Streamlit app title
st.title("🖊️ Handwritten Digit Recognition (0–9)")

st.write("Upload a handwritten digit image (28x28 grayscale) and the model will predict it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read and preprocess image
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    image = image.resize((28, 28))  # resize to 28x28
    img_array = np.array(image) / 255.0  # normalize (0–1)
    img_array = img_array.reshape(1, 28, 28)  # shape (1,28,28)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write(f"### 🎯 Predicted Digit: {predicted_class}")
    st.bar_chart(prediction[0])  # show probability distribution
