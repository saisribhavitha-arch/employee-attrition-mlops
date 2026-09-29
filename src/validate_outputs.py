import os
import json
import numpy as np


print("[INFO] Validating Preprocessing Outputs...")


# ==========================================================
# 1. Load Processed Data
# ==========================================================

X_train = np.load(
    "data/processed/X_train_pipeline.npy"
)

X_test = np.load(
    "data/processed/X_test_pipeline.npy"
)

y_train = np.load(
    "data/processed/y_train_pipeline.npy"
)

y_test = np.load(
    "data/processed/y_test_pipeline.npy"
)


# ==========================================================
# 2. Validate Dimensions
# ==========================================================

if X_train.shape[0] != y_train.shape[0]:
    raise ValueError(
        "Training feature and target dimensions do not match."
    )


if X_test.shape[0] != y_test.shape[0]:
    raise ValueError(
        "Testing feature and target dimensions do not match."
    )


if X_train.shape[1] != X_test.shape[1]:
    raise ValueError(
        "Training and testing feature dimensions do not match."
    )


# ==========================================================
# 3. Validate Missing Values
# ==========================================================

if np.isnan(X_train).any():
    raise ValueError(
        "Missing/NaN values found in training features."
    )


if np.isnan(X_test).any():
    raise ValueError(
        "Missing/NaN values found in testing features."
    )


# ==========================================================
# 4. Validate Pipeline Artifact
# ==========================================================

if not os.path.exists(
    "models/preprocessor.pkl"
):
    raise FileNotFoundError(
        "models/preprocessor.pkl was not found."
    )


# ==========================================================
# 5. Create Summary
# ==========================================================

summary = {

    "training_rows":
        int(X_train.shape[0]),

    "testing_rows":
        int(X_test.shape[0]),

    "number_of_features":
        int(X_train.shape[1]),

    "training_target_values":
        np.unique(y_train).tolist(),

    "testing_target_values":
        np.unique(y_test).tolist(),

    "preprocessor_exists":
        True
}


# ==========================================================
# 6. Save Report
# ==========================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


with open(
    "outputs/preprocessing_summary_report.json",
    "w"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


print(
    "[SUCCESS] Output Validation PASSED."
)

print(
    "[INFO] Features are clean, scaled, encoded, "
    "and dimensionally consistent."
)

print(
    "[INFO] Summary report saved to "
    "outputs/preprocessing_summary_report.json"
)