# Import libraries
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load model
model = load_model('my_model')

# Actions (same as training)
actions = np.array(['BEAUTIFUL','CLASS','COLLEGE','COME','DONT CARE','FRIEND','HELLO','HOW','I','LIKE','PHONE','PROMISE','SLEEP','TAKE CARE','TALK','THANKYOU','TODAY','WANT','WATER','YOU','NO_ACTION'])

# Path to dataset
DATA_PATH = os.path.join('data')

# Parameters
sequence_length = 10

# Lists
X = []
y = []

# Load data
for action_idx, action in enumerate(actions):
    action_path = os.path.join(DATA_PATH, action)

    if not os.path.exists(action_path):
        continue

    for sequence in os.listdir(action_path):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(action_path, sequence, f"{frame_num}.npy")
            if os.path.exists(file_path):
                res = np.load(file_path)
                window.append(res)

        if len(window) == sequence_length:
            X.append(window)
            y.append(action_idx)

# Convert to numpy
X = np.array(X)
y = np.array(y)

print("Total samples loaded:", len(X))

if len(X) == 0:
    print("❌ No data found. Check dataset.")
    exit()

# 🔹 Predictions
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)

# 🔹 Metrics
accuracy = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

print("\n✅ Accuracy:", accuracy)

print("\n✅ Confusion Matrix:")
print(cm)

print("\n✅ Classification Report:")
print(classification_report(y, y_pred, target_names=actions))