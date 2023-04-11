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

cfg_model_path = "last.pt"

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', force_reload=True)

# Function to run YOLOv5 on an image and return the results
def run_yolov5_on_image(image):
    results = model(image)
    return results.render()
 
def videoInput(device, src):
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    if uploaded_video != None:

        ts = datetime.timestamp(datetime.now())
        imgpath = os.path.join('data/uploads', str(ts)+uploaded_video.name)
        outputpath = os.path.join('data/video_output', os.path.basename(imgpath))

        with open(imgpath, mode='wb') as f:
            f.write(uploaded_video.read())  # save video to disk

        st_video = open(imgpath, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
        detect(weights=cfg_model_path, source=imgpath, device=0) if device == 'cuda' else detect(weights=cfg_model_path, source=imgpath, device='cpu')
        st_video2 = open(outputpath, 'rb')
        video_bytes2 = st_video2.read()
        st.video(video_bytes2)
        st.write("Model Prediction")
  
# Create Streamlit app
st.title("Helmet and Number Plate Detection")

# Ask the user if they want to upload an image, video or use the webcam
file_type = st.selectbox("Select file type", ["Image", "Video"])

if file_type == "Image":
    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the image file and run YOLOv5 on it
        image = Image.open(BytesIO(uploaded_file.read()))
        results = run_yolov5_on_image(image)

        # Show the results
        st.image(np.squeeze(results), use_column_width=True)

elif file_type == "Video":
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = True, index=0)
    datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'])
    videoInput(deviceoption, datasrc)
