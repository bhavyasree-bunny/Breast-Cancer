import pandas as pd
import sqlite3

df = pd.read_csv(
    "data/metadata/patients_250.csv"
)

conn = sqlite3.connect(
    "breast_cancer.db"
)

df.to_sql(
    "patients",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Patients imported successfully!")