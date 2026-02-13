import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.camera_input()

import webbrowser
import joblib
import random
import numpy as np
import sqlite3
from datetime import datetime
from collections import Counter
from deepface import DeepFace

# ==========================================
# Create Image Folder Automatically
# ==========================================
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")

# ==========================================
# Create Database Automatically
# ==========================================
conn = sqlite3.connect("emotion_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS emotion_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emotion TEXT,
    image_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

# ==========================================
# Load All Trained Models
# ==========================================
print("Loading trained models...")

rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")
lr_model = joblib.load("lr_model.pkl")
dt_model = joblib.load("dt_model.pkl")
knn_model = joblib.load("knn_model.pkl")

print("All models loaded successfully")

# ==========================================
# Emotion → Music Mapping
# ==========================================
music_map = {
    "happy": "https://www.youtube.com/results?search_query=happy+songs",
    "sad": "https://www.youtube.com/results?search_query=sad+lofi+songs",
    "angry": "https://www.youtube.com/results?search_query=calm+music",
    "neutral": "https://www.youtube.com/results?search_query=focus+music",
    "surprise": "https://www.youtube.com/results?search_query=party+songs",
    "fear": "https://www.youtube.com/results?search_query=relaxing+music",
    "disgust": "https://www.youtube.com/results?search_query=classical+music",
    "sleepy": "https://www.youtube.com/results?search_query=energetic+songs"
}

# ==========================================
# Emotion → Activity Mapping
# ==========================================
activity_map = {
    "happy": ["Smile and enjoy your day.", "Celebrate small wins."],
    "sad": ["Take rest.", "Talk to someone you trust."],
    "angry": ["Take deep breaths.", "Go for a short walk."],
    "neutral": ["Stay focused.", "Drink water."],
    "surprise": ["Enjoy the moment.", "Share excitement."],
    "fear": ["Stay calm.", "Think positive."],
    "disgust": ["Take a short break.", "Organize your space."],
    "sleepy": ["Stretch your body.", "Take a short nap."]
}

# ==========================================
# Load Face Detector
# ==========================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
current_emotion = "neutral"

print("======================================")
print("FINAL EMOTION MUSIC SYSTEM STARTED")
print("Press M → Capture + Save + Music")
print("Press Q → Quit")
print("======================================")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        try:
            # Feature Extraction
            embedding = DeepFace.represent(
                face_img,
                model_name="VGG-Face",
                enforce_detection=False
            )

            features = np.array(embedding[0]["embedding"]).reshape(1, -1)

            # Predictions from All Models
            rf_pred = rf_model.predict(features)[0]
            svm_pred = svm_model.predict(features)[0]
            lr_pred = lr_model.predict(features)[0]
            dt_pred = dt_model.predict(features)[0]
            knn_pred = knn_model.predict(features)[0]

            predictions = [rf_pred, svm_pred, lr_pred, dt_pred, knn_pred]

            # Majority Voting
            vote = Counter(predictions)
            current_emotion = vote.most_common(1)[0][0]

        except:
            pass

        # Draw Box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame,
                    f"Emotion: {current_emotion.upper()}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

    cv2.imshow("Hybrid Emotion System", frame)
    key = cv2.waitKey(1) & 0xFF

    # ==========================
    # When M Pressed
    # ==========================
    if key == ord('m'):

        print("\nFinal Emotion:", current_emotion)

        # Save Image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"{current_emotion}_{timestamp}.jpg"
        image_path = os.path.join("captured_images", image_name)

        cv2.imwrite(image_path, frame)
        print("Image Saved:", image_path)

        # Save To Database
        conn = sqlite3.connect("emotion_data.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO emotion_logs (emotion, image_path)
            VALUES (?, ?)
        """, (current_emotion, image_path))

        conn.commit()
        conn.close()

        print("Saved to Database")

        # Activity Suggestion
        if current_emotion in activity_map:
            activity = random.choice(activity_map[current_emotion])
            print("Suggested Activity:", activity)

        # Open Music
        if current_emotion in music_map:
            print("Opening Recommended Music...")
            webbrowser.open(music_map[current_emotion])

        print("=" * 50)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
