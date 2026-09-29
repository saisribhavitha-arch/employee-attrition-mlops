import os
import json
import mlflow
from mlflow.tracking import MlflowClient


print("Starting Model Registry and Lifecycle...")


# ==========================================================
# 1. Load Best Model Information
# ==========================================================

BEST_MODEL_FILE = (
    "outputs/best_model.json"
)

with open(
    BEST_MODEL_FILE,
    "r"
) as file:

    best_model_info = json.load(file)


best_model = best_model_info["best_model"]

print(
    f"[INFO] Best model selected: {best_model}"
)


# ==========================================================
# 2. Configure MLflow
# ==========================================================

mlflow.set_tracking_uri(
    "sqlite:///artifact/mlflow.db"
)

client = MlflowClient()


# ==========================================================
# 3. Find Latest Run for Best Model
# ==========================================================

experiment = client.get_experiment_by_name(
    "Employee_Attrition_Experiment"
)

if experiment is None:

    raise RuntimeError(
        "MLflow experiment was not found."
    )


runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=[
        "start_time DESC"
    ]
)


best_run = None

for run in runs:

    run_model_name = run.data.params.get(
        "model_name"
    )

    if run_model_name == best_model:

        best_run = run

        break


if best_run is None:

    raise RuntimeError(
        "Could not find MLflow run "
        "for the best model."
    )


run_id = best_run.info.run_id


print(
    f"[INFO] Best model run ID: {run_id}"
)


# ==========================================================
# 4. Register Model
# ==========================================================

registered_model_name = (
    "EmployeeAttritionModel"
)


model_uri = (
    f"runs:/{run_id}/model"
)


print(
    "[INFO] Registering model..."
)


registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name
)


print(
    "[SUCCESS] Model registered successfully!"
)


print(
    f"[INFO] Model version: "
    f"{registered_model.version}"
)


# ==========================================================
# 5. Save Registry Information
# ==========================================================

registry_info = {

    "model_name":
        registered_model_name,

    "model_version":
        registered_model.version,

    "source_run_id":
        run_id,

    "best_model":
        best_model,

    "selection_metric":
        best_model_info[
            "selection_metric"
        ],

    "f1_score":
        best_model_info[
            "f1_score"
        ],

    "lifecycle_stage":
        "Development"

}


os.makedirs(
    "outputs",
    exist_ok=True
)


with open(
    "outputs/model_registry.json",
    "w"
) as file:

    json.dump(
        registry_info,
        file,
        indent=4
    )


print(
    "[SUCCESS] Registry information saved."
)


# ==========================================================
# 6. Lifecycle Information
# ==========================================================

print("\n--- Model Lifecycle ---")

print(
    "Development → Staging → Production"
)

print(
    "[SUCCESS] Model is currently registered "
    "for Development."
)

print(
    "\n[SUCCESS] Model Registry setup completed!"
)