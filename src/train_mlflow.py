import os
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


print("\n--- Starting MLflow Run: Employee Attrition Models ---")


# ==========================================================
# 1. Load Data
# ==========================================================

X_train = np.load(
    "data/processed/X_train.npy"
)

y_train = np.load(
    "data/processed/y_train.npy"
)

X_test = np.load(
    "data/processed/X_test.npy"
)

y_test = np.load(
    "data/processed/y_test.npy"
)


# ==========================================================
# 2. MLflow Configuration
# ==========================================================

os.makedirs(
    "artifact",
    exist_ok=True
)

mlflow.set_tracking_uri(
    "sqlite:///artifact/mlflow.db"
)

mlflow.set_experiment(
    "Employee_Attrition_Experiment"
)


# ==========================================================
# 3. Models
# ==========================================================

model_names = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "svm",
    "naive_bayes"
]


# ==========================================================
# 4. Track Each Model
# ==========================================================

for name in model_names:

    print(
        f"\n[INFO] Starting MLflow run: {name}"
    )

    model = joblib.load(
        f"models/{name}.pkl"
    )

    with mlflow.start_run(
        run_name=name
    ):

        # --------------------------------------------------
        # Model Parameters
        # --------------------------------------------------

        mlflow.log_param(
            "model_name",
            name
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "training_samples",
            len(X_train)
        )

        mlflow.log_param(
            "testing_samples",
            len(X_test)
        )


        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        y_pred = model.predict(
            X_test
        )


        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )


        # --------------------------------------------------
        # ROC-AUC
        # --------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            y_probability = (
                model.predict_proba(X_test)[:, 1]
            )

            roc_auc = roc_auc_score(
                y_test,
                y_probability
            )

        else:

            roc_auc = 0.0


        # --------------------------------------------------
        # Log Metrics
        # --------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )


        # --------------------------------------------------
        # Log Model
        # --------------------------------------------------

        mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=[
        "sklearn.tree._tree.Tree"
    ]
)


        print(
            f"Metrics logged: "
            f"F1 = {f1:.4f} | "
            f"ROC-AUC = {roc_auc:.4f}"
        )


    print(
        f"Run '{name}' successfully tracked!"
    )


print(
    "\n[SUCCESS] All models successfully tracked with MLflow!"
)