import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model_paths_classification = {
    "vgg16_model": './models/aiornot/vgg16_val.h5',
    "resnet50": './models/aiornot/resnet50.h5',
    "inception": './models/aiornot/inceptionv3_.h5',
}

model_paths_severity = {
    "sev_vgg16": './models/sev_models/best_vgg16_model.h5',
    "sev_resnet50": './models/sev_models/ResNet50_sev (1).h5',
    "sev_inception": './models/sev_models/inceptionv3_sev.h5',
}

class_names = ['Fake', 'Real']
severity_class_mapping = {0: 'minor', 1: 'moderate', 2: 'severe'}

def predict_image_vgg(img_array, selected_model):
    try:
        if img_array is None:
            return "No image provided."

        if img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=-1)

        if selected_model == "inception":
            img_array = image.array_to_img(img_array)
            img_array = img_array.resize((75, 75))
            img_array = image.img_to_array(img_array) / 255.0
        else:
            img_array = image.array_to_img(img_array)
            img_array = img_array.resize((32, 32))
            img_array = image.img_to_array(img_array) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        model_path = model_paths_classification.get(selected_model)
        if model_path is None:
            return "Invalid model selection."

        model = load_model(model_path)

        predictions = model.predict(img_array)
        predicted_class = (predictions > 0.51).astype("int32")

        return class_names[predicted_class[0][0]]
    except Exception as e:
        return str(e)

def predict_severity(img_array, selected_model):
    try:
        img = image.array_to_img(img_array)
        if selected_model == "sev_inception":
            img = img.resize((299, 299))  
        else:
            img = img.resize((224, 224))  
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255. 

        model_path = model_paths_severity.get(selected_model)
        if model_path is None:
            return "Invalid model selection."

        model = load_model(model_path)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_severity = severity_class_mapping[predicted_class_index]

        return f"Predicted severity for {selected_model}: {predicted_severity}"

    except Exception as e:
        return str(e)

def st_image_to_array(image_streamlit):
    image_pil = Image.open(image_streamlit)
    image_array = np.array(image_pil)
    return image_array

st.title("Image Classification and Severity Prediction App")
col1, col2 = st.columns([3, 2])
# Task selection
selected_task = col1.radio("Select Task", ["Image Classification", "Severity Prediction"])

# Image classification section
if selected_task == "Image Classification":
    col1.subheader("Image Classification")
    selected_model_classification = col1.selectbox("Select Model for Classification", list(model_paths_classification.keys()))
    image_classification = col1.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])
    if image_classification is not None:
        if col1.button("Submit for Classification"):
            image_array_classification = st_image_to_array(image_classification)
            col2.image(image_array_classification, use_column_width=True)

            result_classification = predict_image_vgg(image_array_classification, selected_model_classification)
            col2.info(result_classification)

# Severity prediction section
elif selected_task == "Severity Prediction":
    col1.subheader("Severity Prediction")
    selected_model_severity = col1.selectbox("Select Model for Severity Prediction", list(model_paths_severity.keys()))
    image_severity = col1.file_uploader("Upload an image for severity prediction", type=["jpg", "jpeg", "png"])
    if image_severity is not None:
        if col1.button("Submit for Severity Prediction"):
            image_array_severity = st_image_to_array(image_severity)
            col2.image(image_array_severity, use_column_width=True)

            result_severity = predict_severity(image_array_severity, selected_model_severity)
            col2.info(result_severity)