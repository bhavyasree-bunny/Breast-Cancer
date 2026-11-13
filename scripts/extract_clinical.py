import requests
import pandas as pd
import json

cases_endpt = "https://api.gdc.cancer.gov/cases"

filters = {
    "op": "=",
    "content": {
        "field": "project.project_id",
        "value": "TCGA-BRCA"
    }
}

params = {
    "filters": json.dumps(filters),
    "format": "JSON",
    "size": 1,
    "expand": "diagnoses"
}

response = requests.get(cases_endpt, params=params)

print("Status:", response.status_code)

hits = response.json()["data"]["hits"]

import pprint

print("Keys in first hit:")
print(hits[0].keys())

print("\nFirst hit:")
pprint.pprint(hits[0])

exit()
rows = []

for case in hits:

    diagnosis = {}

    if case.get("diagnoses"):
        diagnosis = case["diagnoses"][0]

    demographic = case.get("demographic", {})

    rows.append({
        "case_uuid": case.get("case_id"),
        "case_id": case.get("submitter_id"),
        "gender": demographic.get("gender"),
        "race": demographic.get("race"),
        "ethnicity": demographic.get("ethnicity"),
        "age_at_diagnosis": diagnosis.get("age_at_diagnosis"),
        "primary_diagnosis": diagnosis.get("primary_diagnosis"),
        "pathologic_stage": diagnosis.get("tumor_stage")
    })

clinical_df = pd.DataFrame(rows)

clinical_df.to_csv(
    "data/metadata/clinical.csv",
    index=False
)

print("Patients:", len(clinical_df))
print("Saved: data/metadata/clinical.csv")