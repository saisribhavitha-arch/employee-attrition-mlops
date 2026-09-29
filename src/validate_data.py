import pandas as pd


print("[INFO] Starting Data Validation...")


DATA_PATH = (
    "data/raw/"
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

df = pd.read_csv(DATA_PATH)


print(
    f"[INFO] Validating schema "
    f"(Records: {len(df)})..."
)


if df.empty:
    raise ValueError("Dataset is empty.")


if "Attrition" not in df.columns:
    raise ValueError(
        "Target column 'Attrition' is missing."
    )


missing_values = df.isnull().sum().sum()

if missing_values > 0:
    raise ValueError(
        f"Dataset contains "
        f"{missing_values} missing values."
    )


duplicate_rows = df.duplicated().sum()

if duplicate_rows > 0:
    print(
        f"[WARNING] Dataset contains "
        f"{duplicate_rows} duplicate rows."
    )


valid_targets = {"Yes", "No"}

actual_targets = set(
    df["Attrition"].dropna().unique()
)

if not actual_targets.issubset(valid_targets):
    raise ValueError(
        "Unexpected values found "
        "in Attrition column."
    )


print(
    "[SUCCESS] Schema Validation PASSED. "
    "Dataset is clean."
)