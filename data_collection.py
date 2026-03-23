# Import libraries
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from my_functions import image_process, draw_landmarks, keypoint_extraction

# Load trained model
model = load_model('my_model')

# Actions (IMPORTANT: include NO_ACTION)
actions = np.array(['WANT', 'HELLO', 'THANKYOU', 'NO_ACTION'])

# Variables
sequence = []
sentence = []
threshold = 0.85   # increase for more stability

current_word = None
stable_count = 0

# Camera
cap = cv2.VideoCapture(0)

# Mediapipe holistic model
with mp.solutions.holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Process image
        results = image_process(frame, holistic)
        frame = frame.copy()
        draw_landmarks(frame, results)

        # Extract keypoints
        keypoints = keypoint_extraction(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]   # keep last 30 frames

        # Prediction logic
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

            predicted_index = np.argmax(res)
            predicted_word = actions[predicted_index]
            confidence = res[predicted_index]

            # ✅ VALID prediction check
            if confidence > threshold and predicted_word != 'NO_ACTION':

                # 🔒 WORD LOCK SYSTEM
                if predicted_word == current_word:
                    stable_count += 1
                else:
                    current_word = predicted_word
                    stable_count = 1

                # ✅ Only confirm after stable frames
                if stable_count > 15:
                    if len(sentence) == 0 or sentence[-1] != current_word:
                        sentence.append(current_word)

            else:
                # Reset if no valid sign
                current_word = None
                stable_count = 0

        # Limit sentence length
        if len(sentence) > 5:
            sentence = sentence[-5:]

        # Display output
        cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 0), -1)
        cv2.putText(frame,
                    ' '.join(sentence),
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        cv2.imshow('Sign Language Translator', frame)

        # Exit on 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()