import os
import json
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


print("Starting Model Evaluation...")


# ==========================================================
# 1. Load Test Data
# ==========================================================

print("\n[INFO] Loading test data...")

X_test = np.load(
    "data/processed/X_test.npy"
)

y_test = np.load(
    "data/processed/y_test.npy"
)

print("[SUCCESS] Test data loaded!")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ==========================================================
# 2. Model Names
# ==========================================================

model_names = [

    "logistic_regression",

    "decision_tree",

    "random_forest",

    "svm",

    "naive_bayes"

]


# ==========================================================
# 3. Store Results
# ==========================================================

results = {}


# ==========================================================
# 4. Evaluate Each Model
# ==========================================================

for name in model_names:

    print(
        f"\n[INFO] Evaluating {name}..."
    )

    # Load trained model

    model = joblib.load(
        f"models/{name}.pkl"
    )


    # Make predictions

    y_pred = model.predict(
        X_test
    )


    # ------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    # ------------------------------------------------------
    # Precision
    # ------------------------------------------------------

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # ------------------------------------------------------
    # Recall
    # ------------------------------------------------------

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # ------------------------------------------------------
    # F1 Score
    # ------------------------------------------------------

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # ------------------------------------------------------
    # ROC-AUC
    # ------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        y_probability = model.predict_proba(
            X_test
        )[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

    else:

        roc_auc = None


    # ------------------------------------------------------
    # Store Results
    # ------------------------------------------------------

    results[name] = {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        )

    }


    print(
        f"[SUCCESS] {name} evaluation completed!"
    )


# ==========================================================
# 5. Display Evaluation Report
# ==========================================================

print("\n")
print("--- Model Evaluation Report ---")


for name, metrics in results.items():

    print(f"\n{name}")

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1-Score : {metrics['f1_score']:.4f}"
    )

    if metrics["roc_auc"] is not None:

        print(
            f"ROC-AUC  : {metrics['roc_auc']:.4f}"
        )


print("\n-------------------------------")


# ==========================================================
# 6. Identify Best Model
# ==========================================================

best_model = max(
    results,
    key=lambda model:
        results[model]["f1_score"]
)


print(
    f"\n[INFO] Best model based on F1-Score: {best_model}"
)


# ==========================================================
# 7. Save Results
# ==========================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


with open(
    "outputs/evaluation_results.json",
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ==========================================================
# 8. Save Best Model Information
# ==========================================================

best_model_info = {

    "best_model": best_model,

    "selection_metric": "F1-Score",

    "f1_score":
        results[best_model]["f1_score"],

    "accuracy":
        results[best_model]["accuracy"],

    "precision":
        results[best_model]["precision"],

    "recall":
        results[best_model]["recall"],

    "roc_auc":
        results[best_model]["roc_auc"]

}


with open(
    "outputs/best_model.json",
    "w"
) as file:

    json.dump(
        best_model_info,
        file,
        indent=4
    )


print(
    "\n[SUCCESS] Evaluation results saved to outputs/"
)

print(
    "[SUCCESS] Best model information saved!"
)

print(
    "\n[SUCCESS] Model Evaluation completed successfully!"
)