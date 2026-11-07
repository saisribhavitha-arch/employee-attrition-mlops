from common import load_data, prepare_features, split_data, PROC, save_json
import pandas as pd
df=load_data()
X,y=prepare_features(df)
X_train,X_test,y_train,y_test=split_data(df)
X_train.to_csv(PROC/"X_train.csv",index=False); X_test.to_csv(PROC/"X_test.csv",index=False)
y_train.to_csv(PROC/"y_train.csv",index=False); y_test.to_csv(PROC/"y_test.csv",index=False)
save_json(PROC/"preprocessing_metadata.json",{"train_rows":len(X_train),"test_rows":len(X_test),"features":X.columns.tolist(),"target_mapping":{"No":0,"Yes":1},"random_state":42})
print("LAB 3 - PREPROCESSING COMPLETE",X_train.shape,X_test.shape)
