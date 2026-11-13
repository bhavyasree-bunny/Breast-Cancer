import pandas as pd

df = pd.read_csv(
    "data/metadata/slides_clinical.csv"
)

# One row per patient
patient_summary = (
    df.groupby("case_uuid")
    .agg({
        "case_id_x": "first",
        "age_at_diagnosis": "first",
        "primary_diagnosis": "first",
        "pathologic_stage": "first",
        "file_uuid": "count"
    })
    .reset_index()
)

patient_summary.rename(
    columns={
        "case_id_x": "case_id",
        "file_uuid": "number_of_slides"
    },
    inplace=True
)

# Remove missing diagnosis
patient_summary = patient_summary[
    patient_summary["primary_diagnosis"].notna()
]

# Select 250 patients randomly
patients_250 = patient_summary.sample(
    n=250,
    random_state=42
)

patients_250.to_csv(
    "data/metadata/patients_250.csv",
    index=False
)

print("Patients selected:", len(patients_250))
print("Saved: data/metadata/patients_250.csv")