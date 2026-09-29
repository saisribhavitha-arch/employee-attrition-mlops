import os
import json
import numpy as np
import joblib

from sklearn.metrics import f1_score


print("Running Reproducibility Validation...")


# ==========================================================
# 1. Load Test Data
# ==========================================================

X_test = np.load(
    "data/processed/X_test.npy"
)

y_test = np.load(
    "data/processed/y_test.npy"
)


# ==========================================================
# 2. Load Random Forest Model
# ==========================================================

model = joblib.load(
    "models/random_forest.pkl"
)


# ==========================================================
# 3. Execution 1
# ==========================================================

prediction_1 = model.predict(
    X_test
)

f1_1 = f1_score(
    y_test,
    prediction_1,
    zero_division=0
)


# ==========================================================
# 4. Execution 2
# ==========================================================

prediction_2 = model.predict(
    X_test
)

f1_2 = f1_score(
    y_test,
    prediction_2,
    zero_division=0
)


# ==========================================================
# 5. Validate
# ==========================================================

print(
    f"Execution 1 F1: {f1_1:.6f}"
)

print(
    f"Execution 2 F1: {f1_2:.6f}"
)


if np.array_equal(
    prediction_1,
    prediction_2
):

    print(
        "SUCCESS: Pipeline is 100% reproducible."
    )

    reproducible = True

else:

    print(
        "ERROR: Pipeline is not reproducible."
    )

    reproducible = False


# ==========================================================
# 6. Save Report
# ==========================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


report = {

    "execution_1_f1":
        float(f1_1),

    "execution_2_f1":
        float(f1_2),

    "predictions_identical":
        reproducible,

    "random_state":
        42
}


with open(
    "outputs/reproducibility_report.json",
    "w"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


print(
    "Report saved to outputs/reproducibility_report.json"
)