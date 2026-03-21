# %%

# Import necessary libraries
import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import *

# Define the actions (signs)
actions = np.array(['WANT'])

# Define sequences and frames
sequences = 30
frames = 10

# Dataset path
PATH = os.path.join('data')

# Create folders
for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass

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

    for action, sequence, frame in product(actions, range(sequences), range(frames)):

        # WAIT FOR SPACE (only first frame)
        if frame == 0:
            while True:
                ret, image = cap.read()
                if not ret:
                    continue

                results = image_process(image, holistic)
                image = image.copy()  # FIX
                draw_landmarks(image, results)

                cv2.putText(image,
                            f'Recording "{action}" | Sequence {sequence}',
                            (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

                cv2.putText(image,
                            'Press SPACE to start',
                            (20, 450),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

                cv2.imshow('Camera', image)

                # SPACE key detection
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break

                # Window closed
                if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                    break

        # RECORD FRAMES
        else:
            ret, image = cap.read()
            if not ret:
                continue

            results = image_process(image, holistic)
            image = image.copy()  # FIX
            draw_landmarks(image, results)

            cv2.putText(image,
                        f'Recording "{action}" | Sequence {sequence}',
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

            cv2.imshow('Camera', image)
            cv2.waitKey(1)

        # Window closed
        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

        # Save keypoints
        keypoints = keypoint_extraction(results)
        frame_path = os.path.join(PATH, action, str(sequence), str(frame))
        np.save(frame_path, keypoints)

# Release resources
cap.release()
cv2.destroyAllWindows()