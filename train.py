import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# ================================
# IMPORT LIBRARIES
# ================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import joblib
from deepface import DeepFace

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# ================================
# DATASET
# ================================

DATASET_PATH = "dataset"

EMOTIONS = ["happy","sad","angry","neutral","surprise","fear","disgust","sleepy"]

X = []
y = []

print("HYBRID 5-ALGORITHM TRAINING STARTED")

# ================================
# FEATURE EXTRACTION (CNN)
# ================================

for emotion in EMOTIONS:

    emotion_folder = os.path.join(DATASET_PATH, emotion)

    for img_name in os.listdir(emotion_folder):

        img_path = os.path.join(emotion_folder, img_name)

        try:
            img = cv2.imread(img_path)

            embedding = DeepFace.represent(
                img,
                model_name="VGG-Face",
                enforce_detection=False
            )

            features = embedding[0]["embedding"]

            X.append(features)
            y.append(emotion)

        except:
            pass

X = np.array(X)
y = np.array(y)

print("Feature Extraction Done")
print("Total Samples:", len(X))

# ================================
# TRAIN TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 1️⃣ RANDOM FOREST
# ================================

rf_model = RandomForestClassifier(n_estimators=150)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print("Random Forest Accuracy:", rf_acc)
joblib.dump(rf_model, "models/rf_model.pkl")

# ================================
# 2️⃣ SVM
# ================================

svm_model = SVC(kernel="rbf", probability=True)
svm_model.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm_model.predict(X_test))
print("SVM Accuracy:", svm_acc)
joblib.dump(svm_model, "models/svm_model.pkl")

# ================================
# 3️⃣ LOGISTIC REGRESSION
# ================================

lr_model = LogisticRegression(max_iter=2000)
lr_model.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))
print("Logistic Regression Accuracy:", lr_acc)
joblib.dump(lr_model, "models/lr_model.pkl")

# ================================
# 4️⃣ DECISION TREE
# ================================

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
print("Decision Tree Accuracy:", dt_acc)
joblib.dump(dt_model, "models/dt_model.pkl")

# ================================
# 5️⃣ KNN
# ================================

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn_model.predict(X_test))
print("KNN Accuracy:", knn_acc)
joblib.dump(knn_model, "models/knn_model.pkl")

print("ALL MODELS TRAINED SUCCESSFULLY")
