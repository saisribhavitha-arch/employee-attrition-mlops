from common import load_data, train_best, ROOT, save_json, REPORTS
import pandas as pd, json
from datetime import datetime, timezone
df=load_data()
reference=df.drop(columns=["Attrition"])
# Demonstration batch: deterministic sample with a small shift on MonthlyIncome
batch=reference.sample(n=min(200,len(reference)),random_state=7).copy()
batch["MonthlyIncome"]=(batch["MonthlyIncome"]*1.08).round()
drift={}
for col in reference.select_dtypes(include="number").columns:
    a=float(reference[col].mean()); b=float(batch[col].mean())
    drift[col]={"reference_mean":a,"batch_mean":b,"relative_change":float(abs(b-a)/(abs(a)+1e-9))}
flagged=[k for k,v in drift.items() if v["relative_change"]>0.05]
name,model,metrics,_,_=train_best()
report={"timestamp":datetime.now(timezone.utc).isoformat(),"batch_rows":len(batch),"drift_threshold":0.05,"drifted_features":flagged,"drift":drift,"candidate_model":name,"candidate_f1":metrics["f1"],"action":"candidate trained; review before production promotion"}
save_json(REPORTS/"monitoring_report.json",report)
save_json(ROOT/"outputs"/"retraining_report.json",{"candidate_model":name,"candidate_f1":metrics["f1"],"status":"trained"})
print("LAB 12 - MONITORING / DRIFT / RETRAINING")
print("Drifted features:",flagged)
print("Candidate F1:",round(metrics["f1"],4))
