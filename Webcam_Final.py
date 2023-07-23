#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import datetime





def video_bmi():
    '''
    This function predicts the BMI of an individual based on a frame of the camera and overlays it on that video frame. Your camera should open when it's run
    In order to get a prediction, click on the frame of the camera and press the P key. This should overlay the prediction. You can do this repeatedly
    Press the Q key to quit the application
    
    Parameters: 
    mod- The model used to predict the BMI
    
    Output: 
    df- A dataframe that has the date, time and BMI prediction
    
    '''
    pred_bmi = []
    dates = []
    times = []
    
    model_path = 'extra_extra_train_64.h5'
    model = load_model(model_path)
    
    # Load the Haar Cascade classifier for bounding box creation
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    cap = cv2.VideoCapture(0)
    predict_bmi = False  # Flag to indicate when to predict BMI
    bmi = None  # Variable to store the predicted BMI

    while True:
        # Capture a frame from the video stream
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over the detected faces
        for (x, y, w, h) in faces:
            # Draw bounding boxes around the faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not predict_bmi and bmi is not None:
            # Create BMI text overlay
            bmi_text = f"BMI: {bmi}"
            cv2.putText(frame, bmi_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('frame', frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Exit if the user presses the 'q' key
            break
        elif key == ord('p') and not predict_bmi:  # Set the flag to predict BMI when the user presses the 'p' key
            # Resize the frame to (224, 224)
            frame1 = cv2.resize(frame, (224, 224))

            # Convert the frame from BGR to RGB color format
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

            # Convert the frame to a float32 numpy array
            x = np.array(frame1).astype('float32')

            # Normalize the pixel values to be between 0 and 1
            x /= 255.

            # Add an extra dimension to the array
            x = np.expand_dims(x, axis=0)

            # Perform BMI prediction
            predict = model.predict(x, verbose=0)
            bmi = predict[0][0]
            pred_bmi.append(bmi)
            
            current_datetime = datetime.datetime.now()

            # Extract the date and time components
            date = current_datetime.date()
            time = current_datetime.time()
            
            dates.append(date)
            times.append(time)
            

            # Print BMI
            #print(f"BMI: {bmi}")

            # Set the flag to prevent further BMI predictions
            #predict_bmi = True

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()
    
    df = pd.concat([pd.Series(dates, name='Date'), pd.Series(times, name='Time'), pd.Series(pred_bmi, name='BMI')], axis=1)
    return df

