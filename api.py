from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("artifacts/model.pkl")
preprocess = joblib.load("artifacts/preprocess.pkl")

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    X = preprocess.transform(df)
    prob = model.predict_proba(X)[0, 1]
    return {"churn_probability": float(prob)}
