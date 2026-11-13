import sqlite3

conn = sqlite3.connect("breast_cancer.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    case_uuid TEXT PRIMARY KEY,
    case_id TEXT,
    age INTEGER,
    primary_diagnosis TEXT,
    pathologic_stage TEXT
)
""")

conn.commit()
conn.close()

print("Database created successfully!")