import subprocess
import sys


print("[INFO] =========================================")
print("[INFO] Starting Lab 5: Production Data Pipeline")
print("[INFO] =========================================")


# ==========================================================
# 1. Validate Raw Data
# ==========================================================

print("\n[INFO] ---> Executing src/validate_data.py...")

subprocess.run(
    [sys.executable, "src/validate_data.py"],
    check=True
)


# ==========================================================
# 2. Production Preprocessing Pipeline
# ==========================================================

print("\n[INFO] ---> Executing src/preprocess_pipeline.py...")

subprocess.run(
    [sys.executable, "src/preprocess_pipeline.py"],
    check=True
)


# ==========================================================
# 3. Validate Preprocessing Outputs
# ==========================================================

print("\n[INFO] ---> Executing src/validate_outputs.py...")

subprocess.run(
    [sys.executable, "src/validate_outputs.py"],
    check=True
)


# ==========================================================
# 4. Lab 5 Complete
# ==========================================================

print("\n[SUCCESS] Lab 5 Production Pipeline fully executed!")