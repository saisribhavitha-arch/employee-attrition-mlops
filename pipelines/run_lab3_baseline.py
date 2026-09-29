import subprocess
import sys
import os


print("[INFO] =========================================")
print("[INFO] Starting Lab 3: Baseline ML Pipeline")
print("[INFO] =========================================")


# ==========================================================
# 1. Run Preprocessing
# ==========================================================

print("\n[INFO] ---> Executing src/preprocess.py...")

subprocess.run(
    [sys.executable, "src/preprocess.py"],
    check=True
)


# ==========================================================
# 2. Run Model Training
# ==========================================================

print("\n[INFO] ---> Executing src/train.py...")

subprocess.run(
    [sys.executable, "src/train.py"],
    check=True
)


# ==========================================================
# 3. Run Model Evaluation
# ==========================================================

print("\n[INFO] ---> Executing src/evaluate.py...")

subprocess.run(
    [sys.executable, "src/evaluate.py"],
    check=True
)


# ==========================================================
# 4. Lab 3 Complete
# ==========================================================

print("\n[SUCCESS] Lab 3 Pipeline fully executed!")