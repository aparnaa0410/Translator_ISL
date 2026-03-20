# %%

# Import necessary libraries
import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
from tensorflow.keras.models import load_model

# Set the path to the data directory
PATH = os.path.join('data')

# Load action labels
actions = np.array(os.listdir(PATH))

# Load trained model
model = load_model('my_model')

# Initialize variables
sentence = []
keypoints = []
last_prediction = None

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Mediapipe model
with mp.solutions.holistic.Holistic(
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
) as holistic:

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            continue

        # Process frame
        results = image_process(image, holistic)
        image = image.copy()  # FIX read-only issue
        draw_landmarks(image, results)

        # Extract keypoints
        keypoints.append(keypoint_extraction(results))

        # When 10 frames collected → predict
        if len(keypoints) == 10:
            kp_array = np.array(keypoints)
            prediction = model.predict(kp_array[np.newaxis, :, :], verbose=0)
            keypoints = []

            # Confidence threshold
            if np.max(prediction) > 0.9:
                predicted_word = actions[np.argmax(prediction)]

                # Avoid repeating same prediction
                if predicted_word != last_prediction:
                    sentence.append(predicted_word)
                    last_prediction = predicted_word

        # Limit sentence length
        if len(sentence) > 7:
            sentence = sentence[-7:]

        # Reset on SPACE key
        if cv2.waitKey(1) & 0xFF == ord(' '):
            sentence = []
            keypoints = []
            last_prediction = None

        # Capitalize first word
        if sentence:
            sentence[0] = sentence[0].capitalize()

        # Combine letters into words (optional logic)
        if len(sentence) >= 2:
            if sentence[-1] in string.ascii_letters:
                if sentence[-2] in string.ascii_letters:
                    sentence[-1] = sentence[-2] + sentence[-1]
                    sentence.pop(-2)
                    sentence[-1] = sentence[-1].capitalize()

        # Display sentence
        text = ' '.join(sentence)

        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_X = (image.shape[1] - textsize[0]) // 2

        cv2.putText(image, text, (text_X, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Show camera
        cv2.imshow('Camera', image)

        # Exit if window closed
        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()