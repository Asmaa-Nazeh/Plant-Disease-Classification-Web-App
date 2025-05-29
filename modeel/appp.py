





import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model (change the path according to your model file)

model = load_model(r"C:\Users\Public\modeel\best_model1.keras")

# Class names (modify them according to your dataset)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# App title
st.title("Plant Disease Classification using a Keras Model")
st.write("Upload an image of the plant, and the model will classify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=[".jpg", ".jpeg", ".png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    img = image.load_img(uploaded_file, target_size=(224, 224))  # Resize image to model's input size
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prepare image for prediction
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension for prediction

    # Predict using the model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class with highest probability
    confidence = np.max(predictions)  # Get the confidence of the prediction

    # Display results
    st.write(f"Predicted Class: **{class_names[predicted_class]}**")
    st.write(f"Confidence: {confidence * 100:.2f}%")

    # Optionally, you can add a visual representation of the confidence score
    st.bar_chart([confidence * 100])  # Display confidence as a bar chart
