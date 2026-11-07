from common import load_data, prepare_features, PROC
X,y=prepare_features(load_data())
X.assign(Attrition=y).to_csv(PROC/"engineered_employee_data.csv",index=False)
print("LAB 4 - FEATURE ENGINEERING")
print("Added: TenureBand, IncomeBand")
print(X[["TenureBand","IncomeBand"]].head())
