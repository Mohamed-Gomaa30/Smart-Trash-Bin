import cv2
import numpy as np
import tensorflow as tf
import time

# Configuration
MODEL_PATH = "models/recycle_model.h5"
IMG_SIZE = (224, 224)
CLASSES = ["Paper", "Can", "Plastic"]  # Must match the order in prepare_data.py (alphabetical usually, but we'll check)
# Note: image_dataset_from_directory sorts classes alphabetically by default.
# If our folders are 'can', 'paper', 'plastic', then the order is Can, Paper, Plastic.
# Let's verify this assumption or load class names dynamically if possible.
# For now, we'll assume alphabetical: Can, Paper, Plastic. 
# Wait, my prepare_data.py creates: 'can', 'paper', 'plastic'.
# Alphabetical order: 'can', 'paper', 'plastic'.
CLASS_NAMES = ["Can", "Paper", "Plastic"] 

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return None
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
    return model

def preprocess_frame(frame):
    # Resize to model input size
    img = cv2.resize(frame, IMG_SIZE)
    # Convert to float32
    img = img.astype("float32")
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def main():
    import os
    model = load_trained_model()
    if model is None:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting inference. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        input_data = preprocess_frame(frame)

        # Predict
        start_time = time.time()
        preds = model.predict(input_data, verbose=0)
        end_time = time.time()
        
        # Get result
        score = tf.nn.softmax(preds[0])
        class_idx = np.argmax(score)
        confidence = 100 * np.max(score)
        label = CLASS_NAMES[class_idx]
        
        # Display
        fps = 1 / (end_time - start_time)
        text = f"{label}: {confidence:.2f}% (FPS: {fps:.1f})"
        
        # Draw on frame
        color = (0, 255, 0) if confidence > 70 else (0, 0, 255)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Smart Recycle Bin', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
