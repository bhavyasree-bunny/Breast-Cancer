import requests
import pandas as pd
import json

files_endpt = "https://api.gdc.cancer.gov/files"

filters = {
    "op": "and",
    "content": [
        {
            "op": "=",
            "content": {
                "field": "cases.project.project_id",
                "value": "TCGA-BRCA"
            }
        },
        {
            "op": "=",
            "content": {
                "field": "data_format",
                "value": "SVS"
            }
        }
    ]
}

params = {
    "filters": json.dumps(filters),
    "format": "JSON",
    "size": 2000,
    "fields": ",".join([
        "file_id",
        "file_name",
        "file_size",
        "cases.case_id",
        "cases.submitter_id"
    ])
}

response = requests.get(files_endpt, params=params)

print("Status:", response.status_code)

data = response.json()

hits = data["data"]["hits"]

rows = []

for item in hits:

    case_id = None
    submitter_id = None

    if item.get("cases"):
        case = item["cases"][0]
        case_id = case.get("case_id")
        submitter_id = case.get("submitter_id")

    rows.append({
        "file_uuid": item.get("file_id"),
        "file_name": item.get("file_name"),
        "file_size_bytes": item.get("file_size"),
        "case_uuid": case_id,
        "case_id": submitter_id
    })

slides_df = pd.DataFrame(rows)

print("Slides:", len(slides_df))

slides_df.to_csv(
    "data/metadata/slides.csv",
    index=False
)

print("Saved: data/metadata/slides.csv")