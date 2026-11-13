Project: Breast Cancer Metadata Analysis

Dataset:
TCGA-BRCA (GDC Portal)

Data Extracted:
- Patient metadata
- Clinical information
- Histopathology slide metadata

Cohort:
250 patients

Backend:
FastAPI

Endpoints:
/patients
/patient/{case_uuid}
/stats/diagnosis
/stats/patients
/stats/slides

Planned multimodal model
------------------------
The primary model is pretrained ResNet-50 image features fused with clinical
features, a cosine patient similarity graph, and a Graph Attention Network.
The active predictor is `scripts/patient_graph_model.py`; shared preprocessing and fusion are in `scripts/multimodal_features.py`. See [the implementation guide](docs/multimodal_model.md) for preparation,
pilot extraction, training, and limitations.
