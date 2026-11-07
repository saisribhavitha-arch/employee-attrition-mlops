# Employee Attrition Prediction — MLOps Labs 1–13

This project adapts the same 13-lab MLOps progression used in the customer-churn project to **employee attrition**. The target column is `Attrition` (`Yes`/`No`).

## Dataset
`data/raw/employee_attrition.csv` is a reproducible, synthetic educational HR dataset with IBM-HR-style columns. It is not an official IBM HR Analytics dataset and should not be used for real HR decisions. It contains 1,500 rows and is generated with a fixed seed.

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv venv312
.\venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Run commands from the project root. If PowerShell blocks activation, use:
```powershell
.\venv312\Scripts\activate.bat
```

## Labs and commands
| Lab | Topic | Command |
|---|---|---|
| 1 | Project initialization and dataset loading | `python src/01_data_loading.py` |
| 2 | Data validation | `python src/02_data_validation.py` |
| 3 | Data preprocessing | `python src/03_preprocessing.py` |
| 4 | Feature engineering | `python src/04_feature_engineering.py` |
| 5 | Model training | `python src/05_train_model.py` |
| 6 | Model evaluation | `python src/06_evaluate_model.py` |
| 7 | Experiment tracking (local JSON) | `python src/07_experiment_tracking.py` |
| 8 | Model registry/versioning | `python src/08_model_registry.py` |
| 9 | Deployment preparation | `python src/09_deployment_prep.py` |
| 10 | Prediction API | `uvicorn src.api:app --reload` |
| 11 | Docker packaging | `docker build -t employee-attrition-api .` |
| 12 | Monitoring, drift and retraining | `python src/12_monitoring_retraining.py` |
| 13 | End-to-end automation | `python src/13_end_to_end_pipeline.py` |

## Run all labs in sequence
```powershell
python src/01_data_loading.py
python src/02_data_validation.py
python src/03_preprocessing.py
python src/04_feature_engineering.py
python src/05_train_model.py
python src/06_evaluate_model.py
python src/07_experiment_tracking.py
python src/08_model_registry.py
python src/09_deployment_prep.py
python src/12_monitoring_retraining.py
python src/13_end_to_end_pipeline.py
```
Lab 10 API and Lab 11 Docker are launched separately as shown above.

## API test
Start API in one terminal:
```powershell
uvicorn src.api:app --reload
```
In another terminal:
```powershell
python src/test_api.py
```
Open `http://127.0.0.1:8000/docs`.

## Notes
- All paths are relative to the project root.
- The first run creates processed data, model files, reports, and logs.
- Docker requires Docker Desktop to be running.
- The model is for coursework demonstration only, not employment decisions.
