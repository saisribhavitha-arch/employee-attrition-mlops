from common import ROOT, REPORTS, save_json
import json
metrics_path=REPORTS/"evaluation_report.json"
if not metrics_path.exists():
    from common import train_best
    train_best()
metrics=json.loads(metrics_path.read_text())
manifest={"service":"employee-attrition-api","model_file":"models/employee_attrition_model.pkl","input":"employee feature fields (Attrition excluded)","output":["attrition_prediction","attrition_probability"],"metrics":{k:metrics[k] for k in ["accuracy","precision","recall","f1","roc_auc"]}}
save_json(ROOT/"reports"/"deployment_manifest.json",manifest)
print("LAB 9 - DEPLOYMENT PREPARATION")
print(manifest)
