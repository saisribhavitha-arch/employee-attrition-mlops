from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"models"/"employee_attrition_model.pkl"
app=FastAPI(title="Employee Attrition Prediction API",version="1.0.0")

class Employee(BaseModel):
    Age:int=Field(ge=18,le=100)
    BusinessTravel:str="Travel_Rarely"
    DailyRate:int=1000
    Department:str="Research & Development"
    DistanceFromHome:int=5
    Education:int=3
    EducationField:str="Life Sciences"
    EnvironmentSatisfaction:int=3
    Gender:str="Female"
    HourlyRate:int=65
    JobInvolvement:int=3
    JobLevel:int=2
    JobRole:str="Research Scientist"
    JobSatisfaction:int=3
    MaritalStatus:str="Married"
    MonthlyIncome:int=5000
    MonthlyRate:int=15000
    NumCompaniesWorked:int=2
    OverTime:str="No"
    PercentSalaryHike:int=15
    PerformanceRating:int=3
    RelationshipSatisfaction:int=3
    StockOptionLevel:int=1
    TotalWorkingYears:int=8
    TrainingTimesLastYear:int=3
    WorkLifeBalance:int=3
    YearsAtCompany:int=5
    YearsInCurrentRole:int=3
    YearsSinceLastPromotion:int=1
    YearsWithCurrManager:int=3

def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503,detail="Model not trained. Run python src/05_train_model.py")
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status":"ok","model_exists":MODEL_PATH.exists()}

@app.post("/predict")
def predict(employee:Employee):
    model=get_model()
    row=pd.DataFrame([employee.model_dump()])
    row["TenureBand"]=pd.cut(row["YearsAtCompany"],bins=[-1,1,5,10,50],labels=["New","Early","Mid","Long"]).astype(str)
    row["IncomeBand"]=pd.cut(row["MonthlyIncome"],bins=[-1,3000,6000,10000,30000],labels=["Low","Lower-Mid","Upper-Mid","High"]).astype(str)
    pred=int(model.predict(row)[0])
    prob=float(model.predict_proba(row)[0,1])
    return {"attrition_prediction":"Yes" if pred else "No","attrition_probability":round(prob,4)}
