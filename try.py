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


# Define your custom class names here
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

def main():
    st.title("Object Detection App")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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

if __name__ == "__main__":
    main()
