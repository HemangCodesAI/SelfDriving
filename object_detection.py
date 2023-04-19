import cv2
import torch
import numpy as np
# Load YOLOv5 model with specified weights and configuration
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define the classes to detect
# classes = ['person', 'car', 'dog']

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Run the YOLOv5 model on the frame
    results = model(frame)
    
    # Display the resulting frame
    cv2.imshow('frame', np.squeeze(results.render()))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
