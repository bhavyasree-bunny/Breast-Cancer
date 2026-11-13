from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd

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


@app.get("/stats/stages")
def stage_stats():

    stats = (
        patients["pathologic_stage"]
        .fillna("Unknown")
        .value_counts()
        .to_dict()
    )

    return stats

@app.get("/stats/diagnosis")
def diagnosis_stats():

    stats = (
        patients["primary_diagnosis"]
        .fillna("Unknown")
        .value_counts()
        .to_dict()
    )

    return stats

@app.get("/stats/patients")
def patient_count():

    return {
        "total_patients": len(patients)
    }

@app.get("/patient/{case_uuid}")
def get_patient(case_uuid: str):

    result = patients[
        patients["case_uuid"] == case_uuid
    ]

    if len(result) == 0:
        return {"error": "Patient not found"}

    return result.fillna("").astype(str).to_dict(
        orient="records"
    )[0]

@app.get("/stats/patients")
def patient_count():
    return {
        "total_patients": len(patients)
    }

@app.get("/stats/slides")
def slide_stats():

    total_slides = (
        patients["number_of_slides"]
        .astype(int)
        .sum()
    )

    return {
        "total_patients": len(patients),
        "total_slides": int(total_slides)
    }