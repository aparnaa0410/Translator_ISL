# Import necessary libraries
import numpy as np
import os
import string
import mediapipe as mp
import cv2
import pyttsx3
from my_functions import *
from tensorflow.keras.models import load_model
from grammar_model import convert_gloss_to_sentence

engine = pyttsx3.init()

# Optional settings
engine.setProperty('rate', 150)   # speed
engine.setProperty('volume', 1.0) # volume

# Path
PATH = os.path.join('data')

# Load labels
actions = np.array(os.listdir(PATH))

# Load model
model = load_model('my_model')

# Variables
sentence = []
keypoints = []
last_prediction = None
cooldown = 0

# NLP caching
last_gloss = ""
final_text = ""

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Mediapipe
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
        image = image.copy()
        draw_landmarks(image, results)

        # 🔴 Skip if no hand detected
        if not results.left_hand_landmarks and not results.right_hand_landmarks:
            keypoints = []
        else:
            keypoints.append(keypoint_extraction(results))

        # 🔥 Prediction after 10 frames
        if len(keypoints) == 10:
            kp_array = np.array(keypoints)
            prediction = model.predict(kp_array[np.newaxis, :, :], verbose=0)
            keypoints = []

            confidence = np.max(prediction)
            predicted_word = actions[np.argmax(prediction)]

            # Cooldown + no repeat
            if confidence > 0.9:
                if cooldown == 0 and predicted_word != last_prediction:
                    sentence.append(predicted_word)
                    last_prediction = predicted_word
                    cooldown = 20

        # Reduce cooldown
        if cooldown > 0:
            cooldown -= 1

        # Limit sentence length
        if len(sentence) > 7:
            sentence = sentence[-7:]

        # Capitalize first word
        if sentence:
            sentence[0] = sentence[0].capitalize()

        # 🔥 Generate gloss
        gloss = ' '.join(sentence)

        # 🔥 Call NLP only when sentence changes
        if gloss != last_gloss:
            final_text = convert_gloss_to_sentence(sentence)
            last_gloss = gloss

        # 🎯 Display Gloss
        cv2.putText(image, f"Gloss: {gloss}", (20, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # 🎯 Display Sentence
        cv2.putText(image, f"Sentence: {final_text}", (20, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # Show camera
        cv2.imshow('Camera', image)

        # Controls
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            break

        # Reset
        if key == ord(' '):
            sentence = []
            keypoints = []
            last_prediction = None
            cooldown = 0
            final_text = ""
            last_gloss = ""

        # Window closed
        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()