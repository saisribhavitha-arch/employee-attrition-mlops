import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
steps=["01_data_loading.py","02_data_validation.py","03_preprocessing.py","04_feature_engineering.py","05_train_model.py","06_evaluate_model.py","07_experiment_tracking.py","08_model_registry.py","09_deployment_prep.py","12_monitoring_retraining.py"]
print("="*65)
print("LAB 13 - EMPLOYEE ATTRITION END-TO-END MLOPS PIPELINE")
print("="*65)
for script in steps:
    print("\\n>>>",script)
    subprocess.run([sys.executable,str(ROOT/"src"/script)],cwd=ROOT,check=True)
print("\\nLAB 13 COMPLETED SUCCESSFULLY")
print("Reports:",ROOT/"reports")
