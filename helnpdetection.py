import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import time
from io import BytesIO
from PIL import Image


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/soham/Documents/MITAOE 24/TY Comp 22-23/Sem 6/PA/Lab/helmet_detection/yolov5/runs/train/exp5/weights/last.pt', force_reload=True)

# Function to run YOLOv5 on an image and return the results
def run_yolov5_on_image(image):
    results = model(image)
    return results.render()

# Function to run YOLOv5 on a video stream and return the results
def run_yolov5_on_video():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        # Make detections 
        results = model(frame)

        # Show results
        cv2.imshow('YOLO', np.squeeze(results.render()))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        elif st.session_state.stop_webcam:
            # If the user clicked the "Stop" button, break out of the loop
            break
    cap.release()
    cv2.destroyAllWindows()


# Create Streamlit app
st.title("Helmet and Number Plate Detection)

# Ask the user if they want to upload an image, video or use the webcam
file_type = st.selectbox("Select file type", ["Image", "Video", "Webcam"])

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
    # Allow the user to upload a video
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if uploaded_file is not None:
        # Write the video file to disk
        file_name = str(uuid.uuid4()) + os.path.splitext(uploaded_file.name)[1]
        with open(file_name, "wb") as f:
            f.write(uploaded_file.read())

        # Read the video file and run YOLOv5 on it
        cap = cv2.VideoCapture(file_name)
        while cap.isOpened():
            ret, frame = cap.read()

            # Make detections 
            results = model(frame)

            # Show results
            cv2.imshow('YOLO', np.squeeze(results.render()))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        # Remove the temporary video file
        os.remove(file_name)

elif file_type == "Webcam":
    # Run YOLOv5 on the webcam stream
    st.session_state.stop_webcam = False
    if st.button("Stop"):
        st.session_state.stop_webcam = True
    else:
        run_yolov5_on_video()