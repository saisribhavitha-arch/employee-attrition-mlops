import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


print("[INFO] Starting Sklearn Pipeline Preprocessing...")


# ==========================================================
# 1. Paths
# ==========================================================

DATA_PATH = (
    "data/raw/"
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

PROCESSED_PATH = "data/processed"
MODEL_PATH = "models"


os.makedirs(
    PROCESSED_PATH,
    exist_ok=True
)

os.makedirs(
    MODEL_PATH,
    exist_ok=True
)


# ==========================================================
# 2. Load Dataset
# ==========================================================

df = pd.read_csv(DATA_PATH)


# ==========================================================
# 3. Remove Constant Columns
# ==========================================================

constant_columns = [
    column
    for column in df.columns
    if df[column].nunique() == 1
]

df = df.drop(
    columns=constant_columns
)


# ==========================================================
# 4. Feature Engineering
# ==========================================================

df["TenureGroup"] = pd.cut(
    df["YearsAtCompany"],
    bins=[
        -1,
        2,
        5,
        10,
        20,
        float("inf")
    ],
    labels=[
        "0-2 Years",
        "3-5 Years",
        "6-10 Years",
        "11-20 Years",
        "20+ Years"
    ]
)


df["CurrentRoleTenureRatio"] = (
    df["YearsInCurrentRole"] /
    (df["YearsAtCompany"] + 1)
)


df["PromotionGapRatio"] = (
    df["YearsSinceLastPromotion"] /
    (df["YearsAtCompany"] + 1)
)


# ==========================================================
# 5. Remove Highly Correlated Feature
# ==========================================================

df = df.drop(
    columns=["MonthlyIncome"]
)


# ==========================================================
# 6. Target Encoding
# ==========================================================

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})


# ==========================================================
# 7. Separate X and y
# ==========================================================

X = df.drop(
    columns=["Attrition"]
)

y = df["Attrition"]


# ==========================================================
# 8. Identify Columns
# ==========================================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object", "category"]
).columns.tolist()


print(
    f"[INFO] Numerical features: "
    f"{len(numerical_columns)}"
)

print(
    f"[INFO] Categorical features: "
    f"{len(categorical_columns)}"
)


# ==========================================================
# 9. Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# 10. Create Preprocessing Transformer
# ==========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",
            StandardScaler(),
            numerical_columns
        ),

        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ]
)


# ==========================================================
# 11. Create Pipeline
# ==========================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        )
    ]
)


# ==========================================================
# 12. Fit Training Data
# ==========================================================

print(
    "[INFO] Fitting and transforming training data..."
)

X_train_processed = pipeline.fit_transform(
    X_train
)


# ==========================================================
# 13. Transform Test Data
# ==========================================================

print(
    "[INFO] Transforming test data..."
)

X_test_processed = pipeline.transform(
    X_test
)


# ==========================================================
# 14. Save Processed Data
# ==========================================================

import numpy as np

np.save(
    f"{PROCESSED_PATH}/X_train_pipeline.npy",
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else X_train_processed
)

np.save(
    f"{PROCESSED_PATH}/X_test_pipeline.npy",
    X_test_processed.toarray()
    if hasattr(X_test_processed, "toarray")
    else X_test_processed
)

np.save(
    f"{PROCESSED_PATH}/y_train_pipeline.npy",
    y_train.to_numpy()
)

np.save(
    f"{PROCESSED_PATH}/y_test_pipeline.npy",
    y_test.to_numpy()
)


# ==========================================================
# 15. Save Pipeline
# ==========================================================

joblib.dump(
    pipeline,
    f"{MODEL_PATH}/preprocessor.pkl"
)


print(
    "[SUCCESS] Preprocessing completed successfully!"
)

print(
    "[SUCCESS] Pipeline saved as "
    "models/preprocessor.pkl"
)
