import os
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


print("Starting Model Training...")


# ==========================================================
# 1. Load Processed Training Data
# ==========================================================

print("\n[INFO] Loading processed training data...")

X_train = np.load(
    "data/processed/X_train.npy"
)

y_train = np.load(
    "data/processed/y_train.npy"
)

print("[SUCCESS] Training data loaded!")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# ==========================================================
# 2. Create Models
# ==========================================================

models = {

    "logistic_regression":
        LogisticRegression(
            random_state=42,
            max_iter=1000
        ),

    "decision_tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "random_forest":
        RandomForestClassifier(
            random_state=42
        ),

    "svm":
        SVC(
            probability=True,
            random_state=42
        ),

    "naive_bayes":
        GaussianNB()
}


# ==========================================================
# 3. Create Models Directory
# ==========================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ==========================================================
# 4. Train Each Model
# ==========================================================

for name, model in models.items():

    print(
        f"\n[INFO] Training {name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        model,
        f"models/{name}.pkl"
    )

    print(
        f"[SUCCESS] {name} trained and saved!"
    )


# ==========================================================
# 5. Training Complete
# ==========================================================

print("\n[SUCCESS] Model training completed successfully!")

print("\nSaved models:")

for name in models:

    print(
        f"  - models/{name}.pkl"
    )