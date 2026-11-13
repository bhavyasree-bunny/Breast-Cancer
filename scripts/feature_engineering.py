import pandas as pd

df = pd.read_csv(
    "data/metadata/patients_250.csv"
)

df["diagnosis_group"] = (
    df["primary_diagnosis"]
    .apply(
        lambda x:
        "Ductal"
        if "duct" in str(x).lower()
        else (
            "Lobular"
            if "lobular" in str(x).lower()
            else "Other"
        )
    )
)

print(
    df["diagnosis_group"]
    .value_counts()
)