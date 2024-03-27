import cv2
import time
import tensorflow as tf
from keras.models import load_model
import numpy as np

# Load the image classification model
model = load_model('models/imageclassifierV2.h5')

# Initialize the camera capture
cap = cv2.VideoCapture(0)

# Define the interval for capturing frames (in seconds)
interval = 5

while cap.isOpened():
    ret, frame = cap.read()
    
    # Check if the frame is captured successfully
    if ret:
        # Perform image classification on the captured frame
        resize = tf.image.resize(frame, (256, 256))
        yhat = model.predict(np.expand_dims(resize/255, 0))
        predicted_class = "Infertile" if yhat > 0.5 else "Fertile"
        
        # Display the classification result on the frame
        if predicted_class == "Infertile":
            cv2.putText(frame, "Egg: Infertile", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Egg: Fertile", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame with classification result
        cv2.imshow('Camera', frame)
        
        # Check if the user pressed 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('Error: Failed to capture frame.')
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
