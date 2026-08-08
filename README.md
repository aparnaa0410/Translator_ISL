#  Indian Sign Language (ISL) Translator

An AI-powered **Indian Sign Language (ISL) Sign-to-Text and Speech Translation System** that recognizes hand gestures in real time and converts them into meaningful text. The project uses computer vision and deep learning to make communication easier between sign-language users and people who may not understand sign language.

---

# Project Overview

Communication can be challenging for individuals who primarily use sign language, especially when interacting with people who do not understand it.

To address this problem, we developed an **Indian Sign Language Translator** that captures hand gestures through a webcam, extracts important hand landmarks, classifies the gesture using a trained machine learning/deep learning model, and converts the predicted signs into text.

The project also includes a **grammar correction module** to improve the readability of the predicted text.

The dataset used for training was **custom-created by our team** by collecting samples for the required ISL gestures.

---

# Objectives

- Recognize Indian Sign Language gestures using a webcam.
- Convert recognized gestures into text.
- Perform predictions in real time.
- Build and train a custom gesture-recognition model.
- Improve the readability of predicted sentences using grammar correction.
- Provide an accessible communication aid for sign-language users.

---

# Key Features

- Indian Sign Language gesture recognition
-  Real-time webcam-based prediction
-  Hand landmark detection
-  Machine learning/deep learning based classification
-  Custom dataset collected by the team
-  Sign-to-text conversion
-  Grammar correction
-  Real-time prediction
-  Separate model testing functionality

---

#Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **OpenCV** | Webcam capture and image processing |
| **MediaPipe** | Hand landmark detection and tracking |
| **TensorFlow / Keras** | Machine learning/deep learning model |
| **NumPy** | Numerical and feature processing |
| **LanguageTool** | Grammar correction |
| **Jupyter / Python scripts** | Model development and experimentation |

---

# How the System Works

The overall workflow of the system is:

              Webcam
                 │
                 ▼
          Capture Video Frames
                 │
                 ▼
        Hand Detection using
             MediaPipe
                 │
                 ▼
       Extract Hand Landmarks
                 │
                 ▼
       Preprocess / Normalize
             Features
                 │
                 ▼
        Trained ML/DL Model
                 │
                 ▼
        Gesture Classification
                 │
                 ▼
          Predicted Sign
                 │
                 ▼
          Text Generation
                 │
                 ▼
         Grammar Correction
                 │
                 ▼
          Final Text Output
