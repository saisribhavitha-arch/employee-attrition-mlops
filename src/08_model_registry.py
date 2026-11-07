from common import ROOT, save_json, REPORTS
from datetime import datetime, timezone
import joblib
model_path=ROOT/"models"/"employee_attrition_model.pkl"
if not model_path.exists():
    from common import train_best
    train_best()
registry=ROOT/"models"/"model_registry.json"
data={"model_name":"employee_attrition_classifier","version":"1.0.0","stage":"Production","artifact":str(model_path.relative_to(ROOT)),"registered_at":datetime.now(timezone.utc).isoformat()}
save_json(registry,data)
print("LAB 8 - MODEL REGISTRY")
print(data)
