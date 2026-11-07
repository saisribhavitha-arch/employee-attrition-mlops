from common import load_data, save_json, REPORTS
df=load_data()
issues={"missing_values":df.isna().sum().to_dict(),"duplicate_rows":int(df.duplicated().sum()),"target_values":df["Attrition"].value_counts().to_dict(),"invalid_target_rows":int((~df["Attrition"].isin(["Yes","No"])).sum()),"rows":len(df)}
print("LAB 2 - DATA VALIDATION")
print(issues)
save_json(REPORTS/"validation_report.json",issues)
if issues["invalid_target_rows"] or issues["duplicate_rows"]:
    raise SystemExit("Validation failed: inspect validation_report.json")
