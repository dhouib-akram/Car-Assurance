import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
import os, json, cv2, random
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import requests
car_brand_ctaegory = {
    'Aston Martin': 'Luxury',
    'Mercedes-Benz': 'Luxury',
    'Mini': 'Standard',
    'Tesla': 'Electric',
    'GMC': 'SUV',
    'Alfa Romeo': 'Sport',
    'Studebaker': 'Classic',
    'Suzuki': 'Standard',
    'Peugeot': 'Standard',
    'Genesis': 'Luxury',
    'BMW': 'Luxury',
    'Honda': 'Standard',
    'Chrysler': 'Standard',
    'Mazda': 'Standard',
    'Infiniti': 'Luxury',
    'Land Rover': 'SUV',
    'Dodge': 'Standard',
    'Fiat': 'Standard',
    'Maserati': 'Luxury',
    'Saab': 'Standard',
    'Nissan': 'Standard',
    'Hudson': 'Classic',
    'Lincoln': 'Luxury',
    'Volvo': 'Luxury',
    'Mitsubishi': 'Standard',
    'Oldsmobile': 'Classic',
    'Lexus': 'Luxury',
    'Buick': 'Luxury',
    'Jaguar': 'Luxury',
    'Toyota': 'Standard',
    'Volkswagen': 'Standard',
    'Renault': 'Standard',
    'Citroen': 'Standard',
    'Audi': 'Luxury',
    'Subaru': 'Standard',
    'Cadillac': 'Luxury',
    'Pontiac': 'Standard',
    'Porsche': 'Sport',
    'Daewoo': 'Standard',
    'Bugatti': 'Exotic',
    'Jeep': 'SUV',
    'Ram Trucks': 'Truck',
    'Chevrolet': 'Standard',
    'MG': 'Sport',
    'Hyundai': 'Standard',
    'Ferrari': 'Exotic',
    'Acura': 'Luxury',
    'Kia': 'Standard',
    'Bentley': 'Luxury',
    'Ford': 'Standard',
}

cat_repair_cost = {
    'Luxury': {'Dent': {'Minor': 500, 'Moderate': 1000, 'Severe': 2000},
               'Scratch': {'Minor': 300, 'Moderate': 700, 'Severe': 1500},
               'Crack': {'Minor': 800, 'Moderate': 1200, 'Severe': 2500},
               'Glass Shatter': {'Minor': 1000, 'Moderate': 1800, 'Severe': 3000},
               'Lamp Broken': {'Minor': 600, 'Moderate': 1000, 'Severe': 2000},
               'Tire Flat': {'Minor': 200, 'Moderate': 400, 'Severe': 800}},
    'Standard': {'Dent': {'Minor': 400, 'Moderate': 850, 'Severe': 1600},
                 'Scratch': {'Minor': 200, 'Moderate': 550, 'Severe': 1200},
                 'Crack': {'Minor': 700, 'Moderate': 1050, 'Severe': 2200},
                 'Glass Shatter': {'Minor': 800, 'Moderate': 1500, 'Severe': 2700},
                 'Lamp Broken': {'Minor': 500, 'Moderate': 900, 'Severe': 1800},
                 'Tire Flat': {'Minor': 100, 'Moderate': 200, 'Severe': 400}},
    'Sport': {'Dent': {'Minor': 600, 'Moderate': 1100, 'Severe': 2100},
              'Scratch': {'Minor': 350, 'Moderate': 800, 'Severe': 1600},
              'Crack': {'Minor': 900, 'Moderate': 1300, 'Severe': 2600},
              'Glass Shatter': {'Minor': 1100, 'Moderate': 2000, 'Severe': 3200},
              'Lamp Broken': {'Minor': 700, 'Moderate': 1200, 'Severe': 2300},
              'Tire Flat': {'Minor': 150, 'Moderate': 300, 'Severe': 600}},
    'Electric': {'Dent': {'Minor': 700, 'Moderate': 1200, 'Severe': 2300},
                 'Scratch': {'Minor': 400, 'Moderate': 900, 'Severe': 1800},
                 'Crack': {'Minor': 1000, 'Moderate': 1400, 'Severe': 2700},
                 'Glass Shatter': {'Minor': 1200, 'Moderate': 2200, 'Severe': 3400},
                 'Lamp Broken': {'Minor': 800, 'Moderate': 1300, 'Severe': 2400},
                 'Tire Flat': {'Minor': 180, 'Moderate': 360, 'Severe': 720}},
    'SUV': {'Dent': {'Minor': 500, 'Moderate': 950, 'Severe': 1800},
            'Scratch': {'Minor': 250, 'Moderate': 600, 'Severe': 1300},
            'Crack': {'Minor': 800, 'Moderate': 1100, 'Severe': 2200},
            'Glass Shatter': {'Minor': 900, 'Moderate': 1600, 'Severe': 2800},
            'Lamp Broken': {'Minor': 550, 'Moderate': 1000, 'Severe': 2000},
            'Tire Flat': {'Minor': 120, 'Moderate': 240, 'Severe': 480}},
    'Classic': {'Dent': {'Minor': 300, 'Moderate': 700, 'Severe': 1500},
                'Scratch': {'Minor': 150, 'Moderate': 500, 'Severe': 1100},
                'Crack': {'Minor': 600, 'Moderate': 1000, 'Severe': 2000},
                'Glass Shatter': {'Minor': 700, 'Moderate': 1300, 'Severe': 2500},
                'Lamp Broken': {'Minor': 400, 'Moderate': 800, 'Severe': 1700},
                'Tire Flat': {'Minor': 200, 'Moderate': 400, 'Severe': 800}},
}

API_URL = "https://api-inference.huggingface.co/models/dima806/car_brand_image_detection"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def query(uploaded_file):
    try:
        # Use BytesIO to read the file content as bytes
        file_content = uploaded_file.read()
        response = requests.post(API_URL, headers=headers, data=file_content)
        return response.json()
    except Exception as e:
        st.error(f"An error occurred: {e}")


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
CUSTOM_CLASS_NAMES = ['dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat']

def register_custom_dataset():
    # Here you should register your custom dataset if it hasn't been registered yet.
    # Since you haven't provided how your dataset is loaded,
    # I will assume it is already registered in DatasetCatalog under "my_dataset_train" and "my_dataset_val"
    for d in ["train", "val"]:
        MetadataCatalog.get(f"my_dataset_{d}").set(thing_classes=CUSTOM_CLASS_NAMES)
    return MetadataCatalog.get("my_dataset_train")

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "./model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CUSTOM_CLASS_NAMES)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # Set a low threshold to display more predictions
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg

def load_image(image_path):
    return cv2.imread(image_path)

def predict(image, cfg):
    predictor = DefaultPredictor(cfg)
    return predictor(image)

def filter_predictions_by_confidence(instances, min_confidence, max_confidence):
    confidence_indices = (instances.scores > min_confidence) & (instances.scores <= max_confidence)
    return instances[confidence_indices]

def visualize(image, predictions, metadata):
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
    v = v.draw_instance_predictions(predictions.to("cpu"))
    result_image = v.get_image()[:, :, ::-1]
    return result_image

# def segment():
#     # File uploader
    

    


st.title("Car Assurance App")
col1, col2 = st.columns([3, 2])
# Task selection
selected_task = col1.radio("Select Task", ["Image Classification", "Severity Prediction", "Segmentation","Cost estimation & Detect Car Brand"])

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

elif selected_task == "Segmentation":
    uploaded_file = col1.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
            # Perform inference when the user clicks the "Submit" button
        if st.button("Submit"):
            # Read the uploaded image
            image = Image.open(uploaded_file)
            image_array = np.array(image)

            # Perform object detection
            cfg = setup_cfg()
            metadata = register_custom_dataset()
            outputs = predict(image_array, cfg)

            # Get the highest score to set the threshold
            highest_score = (
                outputs["instances"].scores.max().item()
                if len(outputs["instances"])
                else 0
            )
            threshold = highest_score * (2 / 3)  # Calculate 2/3 of the highest score

            # Filter predictions by confidence
            confidence_instances = filter_predictions_by_confidence(
                outputs["instances"], threshold, 1.0
            )

            # Visualize the predictions
            result_image = visualize(image_array, confidence_instances, metadata)

            # Display the result image
            st.image(result_image, caption="Result Image", use_column_width=True)


elif selected_task == "Cost estimation & Detect Car Brand":

    uploaded_file = col1.file_uploader("Choose an image of your car...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col2.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("Estimate Repair Cost"):
            output = query(uploaded_file)
            car_brand = output[0]['label']
            repair_cost_category = car_brand_ctaegory.get(car_brand, 'Unknown')            
            st.subheader("Car Brand Detection Result:")
            st.info(f"Detected Car Brand: {car_brand}")
            st.info(f"Repair Cost Category: {repair_cost_category}")
            st.subheader("Repair Cost Details:")
            if repair_cost_category != 'Unknown':
                repair_costs = cat_repair_cost.get(repair_cost_category, {})
                for damage_type, damage_costs in repair_costs.items():
                    st.write(f"**{damage_type}**:")
                    for severity, cost in damage_costs.items():
                        st.write(f"- {severity.capitalize()}: ${cost}")
            else:
                st.warning("Repair cost category is unknown for the detected car brand.")