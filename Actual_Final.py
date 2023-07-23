#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import cv2
import numpy as np
import datetime
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image

# Create a global list to store the BMI predictions
prediction_history = []

model_path = "C:\\Users\\tfurr\\OneDrive\\Documents\\School\\UChicago\\Spring 2023\\MSCA 31009 ML\\Final_Project\\extra_extra_train_64.h5"

def detect_faces(frame):
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Draw bounding boxes around the faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

def main():
    # Create the Streamlit app header and description
    st.title("BMI Prediction App")
    st.write("This app predicts BMI based on a single image capture.")

    # Load the Haar Cascade classifier for face detection
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # Load the pre-trained model
    model = load_model(model_path)
    
    uploaded_file = st.file_uploader("Upload Previous BMI history file", type="csv")
    # Create a session state to store the prediction history
    session_state = st.session_state

    # Initialize the prediction history if it doesn't exist in the session state
    if "prediction_history" not in session_state:
        session_state.prediction_history = pd.DataFrame(columns=["Date", "Time", "BMI"])

        
    
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        session_state.prediction_history = df
    # Create a placeholder for the captured image
    captured_image = None

    # Display a button to capture an image
    if st.button("Capture Image"):
        # Create the video capture object
        cap = cv2.VideoCapture(0)

        # Capture a single frame
        ret, frame = cap.read()

        # Detect faces in the captured frame
        frame = detect_faces(frame)

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a PIL image from the frame
        captured_image = Image.fromarray(frame_rgb)

        # Display the captured image in the Streamlit app
        st.image(captured_image, channels="RGB", use_column_width=True)

        # Release the video capture object
        cap.release()

    # Perform BMI prediction if an image has been captured
    if captured_image is not None:
        # Resize the image to (224, 224)
        resized_image = captured_image.resize((224, 224))

        # Convert the image to a numpy array
        img_array = np.array(resized_image)

        # Normalize the pixel values to be between 0 and 1
        img_array = img_array.astype('float32') / 255.

        # Add an extra dimension to the array
        img_array = np.expand_dims(img_array, axis=0)

        # Perform BMI prediction using the loaded model
        prediction = model.predict(img_array)[0][0]
        bmi = prediction

        # Get the current date and time
        current_datetime = datetime.datetime.now()
        date = current_datetime.date()
        time = current_datetime.time()
        
        df1 = pd.DataFrame({"Date":[date], "Time":[time], "BMI": [bmi]})

        # Append the new prediction to the history in session state
        session_state.prediction_history = pd.concat([session_state.prediction_history, df1], axis=0)
        df = pd.DataFrame(session_state.prediction_history)
        
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df(df)
        # Display the BMI prediction history as a table
        st.subheader("BMI Prediction History")
        #st.dataframe(pd.DataFrame(session_state.prediction_history))
        st.dataframe(df)

        
        # Add a button to download the prediction history

            
        st.download_button(
            "Press to Download",
             csv,
            "file.csv",
            "text/csv",
             key='download-csv')
         
           
        

if __name__ == '__main__':
    main()
