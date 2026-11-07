import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


print("Starting Preprocessing Pipeline...")


# ==========================================================
# 1. File Paths
# ==========================================================

DATA_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"

PROCESSED_PATH = "data/processed"
MODEL_PATH = "models"
OUTPUT_PATH = "outputs"


# Create required folders if they do not exist

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)


# ==========================================================
# 2. Load Dataset
# ==========================================================

print("\n[INFO] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("[SUCCESS] Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# ==========================================================
# 3. Check Missing Values
# ==========================================================

print("\n[INFO] Checking missing values...")

missing_values = df.isnull().sum().sum()

print("Total missing values:", missing_values)


# ==========================================================
# 4. Check Duplicate Rows
# ==========================================================

print("\n[INFO] Checking duplicate rows...")

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)


# ==========================================================
# 5. Remove Constant Columns
# ==========================================================

print("\n[INFO] Checking constant columns...")

constant_columns = [
    column
    for column in df.columns
    if df[column].nunique() == 1
]

print("Constant columns:")
print(constant_columns)


df = df.drop(
    columns=constant_columns
)

print(
    "Shape after removing constant features:",
    df.shape
)


# ==========================================================
# 6. Feature Engineering - Tenure Group
# ==========================================================

print("\n[INFO] Creating TenureGroup...")

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

print("TenureGroup created successfully.")


# ==========================================================
# 7. Feature Engineering - Current Role Tenure Ratio
# ==========================================================

print("\n[INFO] Creating CurrentRoleTenureRatio...")

df["CurrentRoleTenureRatio"] = (
    df["YearsInCurrentRole"] /
    (df["YearsAtCompany"] + 1)
)

print("CurrentRoleTenureRatio created successfully.")


# ==========================================================
# 8. Feature Engineering - Promotion Gap Ratio
# ==========================================================

print("\n[INFO] Creating PromotionGapRatio...")

df["PromotionGapRatio"] = (
    df["YearsSinceLastPromotion"] /
    (df["YearsAtCompany"] + 1)
)

print("PromotionGapRatio created successfully.")


# ==========================================================
# 9. Remove Highly Correlated Feature
# ==========================================================

print("\n[INFO] Removing highly correlated feature...")

df = df.drop(
    columns=["MonthlyIncome"]
)

print("Removed feature: MonthlyIncome")

print(
    "Remaining number of features:",
    df.shape[1]
)


# ==========================================================
# 10. Convert Target Variable
# ==========================================================

print("\n[INFO] Converting Attrition into binary form...")

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

print(
    "Attrition classes:",
    df["Attrition"].unique()
)


# ==========================================================
# 11. One-Hot Encoding
# ==========================================================

print("\n[INFO] Performing one-hot encoding...")

categorical_columns = (
    df.drop(
        columns=["Attrition"]
    )
    .select_dtypes(
        include=["object", "category"]
    )
    .columns
)



df_encoded = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

# Convert boolean columns to integers
bool_columns = (
    df_encoded
    .select_dtypes(include="bool")
    .columns
)

df_encoded[bool_columns] = (
    df_encoded[bool_columns]
    .astype(int)
)

# Make sure every feature is numeric
df_encoded = df_encoded.astype(float)



print(
    "Dataset shape after encoding:",
    df_encoded.shape
)


# ==========================================================
# 12. Separate Features and Target
# ==========================================================

print("\n[INFO] Separating features and target...")

X = df_encoded.drop(
    "Attrition",
    axis=1
)

y = df_encoded["Attrition"]


print("Feature shape:", X.shape)

print("Target shape:", y.shape)


# ==========================================================
# 13. Train-Test Split
# ==========================================================

print("\n[INFO] Performing train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(
    "Training data shape:",
    X_train.shape
)

print(
    "Testing data shape:",
    X_test.shape
)

print(
    "Training target shape:",
    y_train.shape
)

print(
    "Testing target shape:",
    y_test.shape
)


# ==========================================================
# 14. Check Class Distribution
# ==========================================================

print("\n[INFO] Checking class distribution...")

print("Before oversampling:")

print(
    y_train.value_counts()
)


# ==========================================================
# 15. Random Oversampling
# ==========================================================

print("\n[INFO] Applying Random Oversampling...")

ros = RandomOverSampler(
    random_state=42
)


X_train_balanced, y_train_balanced = (
    ros.fit_resample(
        X_train,
        y_train
    )
)


print("After oversampling:")

print(
    y_train_balanced.value_counts()
)


# ==========================================================
# 16. Feature Scaling
# ==========================================================

print("\n[INFO] Applying StandardScaler...")

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train_balanced
)

X_test_scaled = scaler.transform(
    X_test
)


print("Feature scaling completed.")


# ==========================================================
# 17. Save Processed Data
# ==========================================================

print("\n[INFO] Saving processed data...")


np.save(
    f"{PROCESSED_PATH}/X_train.npy",
    X_train_scaled
)

np.save(
    f"{PROCESSED_PATH}/X_test.npy",
    X_test_scaled
)

np.save(
    f"{PROCESSED_PATH}/y_train.npy",
    y_train_balanced.to_numpy()
)

np.save(
    f"{PROCESSED_PATH}/y_test.npy",
    y_test.to_numpy()
)


# ==========================================================
# 18. Save Feature Names
# ==========================================================

feature_names = X.columns.tolist()

with open(
    f"{PROCESSED_PATH}/feature_names.json",
    "w"
) as file:

    json.dump(
        feature_names,
        file,
        indent=4
    )


# ==========================================================
# 19. Save Scaler
# ==========================================================

joblib.dump(
    scaler,
    f"{MODEL_PATH}/scaler.pkl"
)


# ==========================================================
# 20. Save Preprocessing Information
# ==========================================================

preprocessing_info = {

    "original_rows": int(df.shape[0]),

    "encoded_features": int(X.shape[1]),

    "training_rows_before_oversampling":
        int(X_train.shape[0]),

    "training_rows_after_oversampling":
        int(X_train_balanced.shape[0]),

    "testing_rows":
        int(X_test.shape[0]),

    "constant_columns_removed":
        constant_columns,

    "correlated_feature_removed":
        "MonthlyIncome",

    "test_size":
        0.20,

    "random_state":
        42,

    "scaling":
        "StandardScaler",

    "oversampling":
        "RandomOverSampler"
}


with open(
    f"{OUTPUT_PATH}/preprocessing_info.json",
    "w"
) as file:

    json.dump(
        preprocessing_info,
        file,
        indent=4
    )


print("\n[SUCCESS] Preprocessing completed successfully!")

print(
    "[SUCCESS] Processed files saved to data/processed/"
)

print(
    "[SUCCESS] Scaler saved to models/scaler.pkl"
)

print(
    "[SUCCESS] Preprocessing report saved to outputs/"
)