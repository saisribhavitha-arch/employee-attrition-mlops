from common import train_best, save_json, REPORTS
name,model,metrics,_,results=train_best()
save_json(REPORTS/"training_comparison.json",results)
print("LAB 5 - MODEL TRAINING")
print("Selected:",name)
print("F1:",round(metrics["f1"],4))
print("Saved: models/employee_attrition_model.pkl")
