from common import train_best, ROOT, save_json
from datetime import datetime, timezone
import json
name,model,metrics,_,_=train_best()
path=ROOT/"outputs"/"experiment_tracking.json"
runs=json.loads(path.read_text()) if path.exists() else []
runs.append({"run_id":len(runs)+1,"timestamp":datetime.now(timezone.utc).isoformat(),"model":name,"metrics":{k:metrics[k] for k in ["accuracy","precision","recall","f1","roc_auc"]},"artifact":"models/employee_attrition_model.pkl"})
save_json(path,runs)
print("LAB 7 - EXPERIMENT TRACKING")
print("Logged run:",runs[-1])
