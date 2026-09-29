import subprocess
import sys


print("[INFO] =========================================")
print("[INFO] Starting Lab 6: Model Registry & Lifecycle")
print("[INFO] =========================================")


# ==========================================================
# 1. Register Best Model
# ==========================================================

print(
    "\n[INFO] ---> Executing src/register_model.py..."
)

subprocess.run(
    [
        sys.executable,
        "src/register_model.py"
    ],
    check=True
)


# ==========================================================
# 2. Complete
# ==========================================================

print(
    "\n[SUCCESS] Lab 6 Model Registry Pipeline "
    "fully executed!"
)