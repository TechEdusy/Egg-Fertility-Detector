import cv2
import numpy as np
import urllib.request
from keras.models import load_model

# Load your pre-trained machine learning model for egg fertility detection
# Replace 'load_model' with the actual method to load your model
model = load_model('models/imageclassifierV2.h5')

def preprocess_image(image):
    """Preprocesses an image for egg fertility detection."""
    # Resize the image to the expected shape (256x256)
    resized_image = cv2.resize(image, (256, 256))
    # Normalize pixel values (assuming 8-bit depth)
    normalized_image = resized_image / 255.0
    # Expand dimensions to match model input shape (add batch dimension)
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

def predict_fertility(image, model):
    """Predicts the fertility of an egg image using a pre-trained model."""
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Perform prediction using the loaded model
    prediction = model.predict(preprocessed_image)
    return prediction

def detect_egg(frame):
    """Detects the presence of eggs in the given frame."""
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for egg color in HSV
    lower_bound = np.array([20, 50, 50])  # Lower bound for yellow color
    upper_bound = np.array([40, 255, 255])  # Upper bound for yellow color
    
    # Threshold the HSV image to get only yellow regions (eggs)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours are found, eggs are present
    if contours:
        return True
    else:
        return False

# URL of the IP camera stream
url = 'http://192.168.111.100/cam-hi.jpg'

while True:
    # Fetch frame from the IP camera stream
    img_resp = urllib.request.urlopen(url)
    img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_np, -1)
    
    # Check if there is an egg in the frame
    egg_present = detect_egg(frame)
    
    if egg_present:
        # Perform fertility prediction on the frame using the pre-trained model
        prediction = predict_fertility(frame, model)
        
        # Display prediction result on the frame
        # Modify this part according to your prediction output
        # For example, display a text indicating fertility status
        if prediction == 1:
            text = "Fertile"
            color = (0, 255, 0)  # Green color for fertile eggs
        else:
            text = "Non-fertile"
            color = (0, 0, 255)  # Red color for non-fertile eggs
    else:
        text = "No egg"
        color = (255, 0, 0)  # Blue color for indicating no egg
    
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Display the frame
    cv2.imshow('Image', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all OpenCV windows
cv2.destroyAllWindows()
