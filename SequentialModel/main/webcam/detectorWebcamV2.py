import cv2
import time
import tensorflow as tf
from keras.models import load_model
import numpy as np
import os

# Load the image classification model
model = load_model('models/imageclassifierV2.h5')

# Initialize the camera capture
cap = cv2.VideoCapture(0)

# Define the interval for capturing frames (in seconds)
interval = 2

# Initialize a counter to keep track of captured images
image_counter = 0

# Directory to save the captured images
save_dir = 'captured_images'

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

while cap.isOpened():
    ret, frame = cap.read()
    
    # Check if the frame is captured successfully
    if ret:
        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Increment the counter and check if it's time to capture a new image
        image_counter += 1
        if image_counter == (interval * cap.get(cv2.CAP_PROP_FPS)):
            # Perform image classification on the captured frame
            resize = tf.image.resize(frame, (256, 256))
            yhat = model.predict(np.expand_dims(resize/255, 0))
            predicted_class = "Infertile" if yhat > 0.5 else "Fertile"
            
            # # Display the result on the frame
            # if predicted_class == "Infertile":
            #     cv2.putText(frame, "Egg detected: Infertile", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # else:
            #     cv2.putText(frame, "No eggs detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if predicted_class == "Infertile":
                    cv2.putText(frame, "Egg: Infertile", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                    cv2.putText(frame, "Egg: Fertile", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save the captured frame as an image
            image_name = f'image_{int(time.time())}.jpg'  # Generate a unique image name using the current timestamp
            cv2.imwrite(os.path.join(save_dir, image_name), frame)
            print(f'Image saved: {os.path.join(save_dir, image_name)}')
            
            # Reset the counter
            image_counter = 0
        
        # Check if the user pressed 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('Error: Failed to capture frame.')
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
