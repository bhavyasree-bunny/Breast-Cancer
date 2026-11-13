from fastapi import FastAPI
import pandas as pd
import json
from fastapi.responses import JSONResponse

app = FastAPI()

patients = pd.read_csv(
    "data/metadata/patients_250.csv"
)

@app.get("/")
def home():
    return {
        "message": "Breast Cancer API"
    }

@app.get("/patients")
def get_patients():

    records = (
        patients
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )

    return JSONResponse(content=records)