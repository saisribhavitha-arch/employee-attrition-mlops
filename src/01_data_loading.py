from common import load_data, save_json, REPORTS
df=load_data()
print("LAB 1 - DATA LOADING")
print("Shape:", df.shape)
print(df.head())
print(df.dtypes)
save_json(REPORTS/"data_loading_report.json", {"rows":len(df),"columns":df.columns.tolist(),"shape":list(df.shape)})
