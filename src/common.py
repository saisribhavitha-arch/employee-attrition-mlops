from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "employee_attrition.csv"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
for p in [PROC, MODELS, OUT, REPORTS]:
    p.mkdir(parents=True, exist_ok=True)

def load_data():
    return pd.read_csv(DATA)

def prepare_features(df):
    data = df.copy()
    y = data.pop("Attrition").map({"No": 0, "Yes": 1})
    # deterministic feature engineering
    data["TenureBand"] = pd.cut(data["YearsAtCompany"], bins=[-1,1,5,10,50], labels=["New","Early","Mid","Long"]).astype(str)
    data["IncomeBand"] = pd.cut(data["MonthlyIncome"], bins=[-1,3000,6000,10000,30000], labels=["Low","Lower-Mid","Upper-Mid","High"]).astype(str)
    return data, y

def build_pipeline(X, model=None):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])
    if model is None:
        model = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    return Pipeline([("preprocess", prep), ("model", model)])

def split_data(df):
    X, y = prepare_features(df)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def metrics_for(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    return {
        "accuracy": float(accuracy_score(y_test,pred)),
        "precision": float(precision_score(y_test,pred,zero_division=0)),
        "recall": float(recall_score(y_test,pred,zero_division=0)),
        "f1": float(f1_score(y_test,pred,zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test,proba)),
        "confusion_matrix": confusion_matrix(y_test,pred).tolist(),
        "classification_report": classification_report(y_test,pred,zero_division=0,output_dict=True)
    }

def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def train_best():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    candidates = [
        ("random_forest", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
        ("logistic_regression", LogisticRegression(max_iter=1500, class_weight="balanced"))
    ]
    best = None
    results = {}
    for name, estimator in candidates:
        pipe = build_pipeline(X_train, estimator)
        pipe.fit(X_train, y_train)
        m = metrics_for(pipe, X_test, y_test)
        results[name] = m
        if best is None or m["f1"] > best[0]:
            best = (m["f1"], name, pipe, m)
    _, name, model, metrics = best
    joblib.dump(model, MODELS/"employee_attrition_model.pkl")
    save_json(REPORTS/"evaluation_report.json", {"best_model":name, **metrics, "rows":len(df)})
    return name, model, metrics, (X_train, X_test, y_train, y_test), results
