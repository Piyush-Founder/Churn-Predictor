from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app =FastAPI()


model =joblib.load('xgboost_churn_model.pkl')
feature_columns =joblib.load('feature_columns.pkl')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
   
   
@app.get('/')
def home(): 
    return {'message': 'Api is running'}



@app.post('/predict')
def predict(data :CustomerData):
    try:
        input_dict = {
            'gender':data.gender,
            'SeniorCitizen':data.SeniorCitizen,
            'Partner': data.Partner,
            'Dependents': data.Dependents,
            'tenure': data.tenure,
            'PhoneService': data.PhoneService,
            'MultipleLines': data.MultipleLines,
            'InternetService': data.InternetService,
            'OnlineSecurity': data.OnlineSecurity,
            'OnlineBackup': data.OnlineBackup,
            'DeviceProtection': data.DeviceProtection,
            'TechSupport': data.TechSupport,
            'StreamingTV': data.StreamingTV,
            'StreamingMovies': data.StreamingMovies,
            'Contract': data.Contract,
            'PaperlessBilling': data.PaperlessBilling,
            'PaymentMethod': data.PaymentMethod,
            'MonthlyCharges': data.MonthlyCharges,
            'TotalCharges': data.TotalCharges
        }
        
        input_df =pd.DataFrame([input_dict])
        input_encoded =pd.get_dummies(input_df,drop_first=True)
        
        input_encoded =input_encoded.reindex(columns=feature_columns,fill_value=0)
        
        prediction =model.predict(input_encoded)[0]
        probability =model.predict_proba(input_encoded)[0][1]
        
        return {
            "prediction": int(prediction),
            "churn": "Yes" if prediction == 1 else "No",
            "probability": float(probability)
}
    
    except Exception as e:
         raise HTTPException(status_code=500,detail=str(e))