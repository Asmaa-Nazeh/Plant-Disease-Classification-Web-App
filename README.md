---

# Plant Disease Classification Web App

## Overview

This project is a web application for classifying **plant leaf diseases** using **deep learning**. It is built with **Streamlit** and uses a model trained on the [**PlantVillage dataset**](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage), which contains images of healthy and diseased leaves from multiple crops.

The system allows users to upload a leaf image and returns the predicted disease class along with the model’s confidence score. The deep learning model was trained using **transfer learning** with several pre-trained architectures. The final application supports over **38 disease classes** from crops such as Tomato, Apple, Potato, Grape, and more.

---

## Features

* Upload an image of a plant leaf for prediction
* Uses deep learning models based on transfer learning
* Displays prediction class and confidence level
* Provides a simple interface via Streamlit

---

## Model Training

Model training is implemented in the `plantvillage2.ipynb` notebook. Several pre-trained models were evaluated:

* VGG16
* ResNet50
* InceptionV3
* MobileNetV2
* EfficientNetB0

### Training Workflow

* Image preprocessing and augmentation using `ImageDataGenerator`
* Fine-tuning models from `tensorflow.keras.applications`
* Evaluation based on validation accuracy
* The best-performing model was saved as `best_model1.keras`

Dataset used: [PlantVillage - Kaggle](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)

---

## Supported Classes

The model supports 38 plant disease classes, including:

* **Tomato**: Bacterial Spot, Early Blight, Late Blight, etc.
* **Apple**: Scab, Black Rot, Cedar Apple Rust
* **Potato**: Early Blight, Late Blight
* **Corn**, **Peach**, **Strawberry**, **Bell Pepper**, **Grape**, **Soybean**, **Squash**, and others.

---

## Project Structure

```
.
├── appp.py                # Streamlit web application
├── plantvillage2.ipynb    # Jupyter Notebook for model training and evaluation
├── best_model1.keras      # Trained Keras model file (manually added)
├── README.md              # Project documentation
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/plant-disease-classification-app.git
cd plant-disease-classification-app
```

### 2. Install Dependencies

```bash
pip install streamlit tensorflow numpy
```

Or install from a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Launch the App

```bash
streamlit run appp.py
```

---

## Example

After uploading a plant leaf image, the model returns the predicted class and confidence score:

```
Predicted Class: Tomato___Late_blight
Confidence: 95.43%
```

A bar chart is also displayed to represent confidence visually.

---

## License

This project is licensed under the **MIT License**.

---


