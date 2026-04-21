from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "xgboost_churn_model.pkl")
feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templetes"))


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


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/predict-page", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.post("/predict")
def predict(data: CustomerData):
    try:
        input_dict = {
            "gender": data.gender,
            "SeniorCitizen": data.SeniorCitizen,
            "Partner": data.Partner,
            "Dependents": data.Dependents,
            "tenure": data.tenure,
            "PhoneService": data.PhoneService,
            "MultipleLines": data.MultipleLines,
            "InternetService": data.InternetService,
            "OnlineSecurity": data.OnlineSecurity,
            "OnlineBackup": data.OnlineBackup,
            "DeviceProtection": data.DeviceProtection,
            "TechSupport": data.TechSupport,
            "StreamingTV": data.StreamingTV,
            "StreamingMovies": data.StreamingMovies,
            "Contract": data.Contract,
            "PaperlessBilling": data.PaperlessBilling,
            "PaymentMethod": data.PaymentMethod,
            "MonthlyCharges": data.MonthlyCharges,
            "TotalCharges": data.TotalCharges,
        }

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        return {
            "prediction": int(prediction),
            "churn": "Yes" if prediction == 1 else "No",
            "probability": float(probability),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
