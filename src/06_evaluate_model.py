from common import train_best, save_json, REPORTS
name,model,metrics,_,_=train_best()
save_json(REPORTS/"model_evaluation.json",{"model":name,**metrics})
print("LAB 6 - MODEL EVALUATION")
for k in ["accuracy","precision","recall","f1","roc_auc"]: print(k,round(metrics[k],4))
print("Confusion matrix:",metrics["confusion_matrix"])
