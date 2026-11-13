import pandas as pd

slides_df = pd.read_csv(
    "data/metadata/slides.csv"
)

clinical_df = pd.read_csv(
    "data/metadata/clinical.csv"
)

combined_df = slides_df.merge(
    clinical_df,
    on="case_uuid",
    how="left"
)

combined_df.to_csv(
    "data/metadata/slides_clinical.csv",
    index=False
)

print("Rows:", len(combined_df))
print("Patients:", combined_df["case_uuid"].nunique())

print("Saved: data/metadata/slides_clinical.csv")