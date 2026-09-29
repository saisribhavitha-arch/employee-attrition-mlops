import subprocess
import sys


print("[INFO] =========================================")
print("[INFO] Starting Lab 4: MLflow Experiment Tracking")
print("[INFO] =========================================")


print("\n[INFO] ---> Executing src/preprocess.py...")

subprocess.run(
    [sys.executable, "src/preprocess.py"],
    check=True
)


print("\n[INFO] ---> Executing src/train_mlflow.py...")

subprocess.run(
    [sys.executable, "src/train_mlflow.py"],
    check=True
)


print("\n[INFO] ---> Executing src/validate_reproducibility.py...")

subprocess.run(
    [sys.executable, "src/validate_reproducibility.py"],
    check=True
)


print("\n[SUCCESS] Lab 4 Pipeline fully executed!")