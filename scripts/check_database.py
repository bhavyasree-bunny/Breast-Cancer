import sqlite3
import pandas as pd

conn = sqlite3.connect(
    "breast_cancer.db"
)

df = pd.read_sql(
    "SELECT * FROM patients LIMIT 10",
    conn
)

print(df)

conn.close()


































