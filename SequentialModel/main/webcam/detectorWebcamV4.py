import cv2
import numpy as np
import keras
from keras.models import load_model

# Load the pre-trained machine learning model for egg fertility detection
model = load_model('models/imageclassifierV2.h5')

def preprocess_image(image):
    """Preprocesses an image for egg fertility detection."""
    # Resize the image to the expected shape (150x150)
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

def detect_egg(frame, model):
    """Detects the presence and fertility of an egg in the given frame."""
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform edge detection
    edges = cv2.Canny(gray, 50, 256)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours are found, there is an egg present
    if contours:
        # Filter contours based on size
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500< area < 2000:  # Adjust these thresholds as needed
                valid_contours.append(contour)
        
        # Get the largest contour (assuming it corresponds to the egg)
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Extract the egg region from the frame
            egg_region = frame[y:y+h, x:x+w]
            
            # Perform fertility prediction on the egg region using the pre-trained model
            prediction = predict_fertility(egg_region, model)
            
            # Determine fertility status based on prediction
            fertility_status = "Fertile" if prediction[0][0] < 0.5 else "Non-fertile"
            
            # Display fertility status on the frame
            color = (0, 255, 0) if fertility_status == "Fertile" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, fertility_status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return True  # Egg is present
        else:
            return False  # No valid egg contours found
    else:
        return False  # No egg contours found


# Initialize webcam
cap = cv2.VideoCapture(0)    

interval = 5

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Check if there is an egg in the frame
    egg_present = detect_egg(frame, model)
    
    # Display message if no egg is detected
    if not egg_present:
        cv2.putText(frame, "No egg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Image', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
