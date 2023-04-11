import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import time
import uuid
from io import BytesIO
from PIL import Image
from datetime import datetime

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', force_reload=True)

# Function to run YOLOv5 on an image and return the results
def run_yolov5_on_image(image):
    results = model(image)
    return results.render()

st.title("Helmet and Number Plate Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read the image file and run YOLOv5 on it
    image = Image.open(BytesIO(uploaded_file.read()))
    results = run_yolov5_on_image(image)

    # Show the results
    st.image(np.squeeze(results), use_column_width=True)
