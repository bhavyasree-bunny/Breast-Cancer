# Breast Cancer Prediction Using Multimodal Deep Learning and Graph Neural Networks

## Overview

This project investigates a multimodal machine learning framework for breast cancer analysis using clinical information and histopathology whole-slide images from The Cancer Genome Atlas Breast Invasive Carcinoma (TCGA-BRCA) dataset.

The proposed framework combines pretrained convolutional neural network image features, clinical features, patient similarity graphs, and Graph Attention Networks (GATs).

The repository also includes a FastAPI backend for accessing patient metadata and dataset statistics.

## Dataset

**Source:** TCGA-BRCA, Genomic Data Commons (GDC) Portal

**Initial cohort:** 250 patients

The project includes:

- Patient metadata
- Clinical information
- Histopathology slide metadata
- Whole-slide histopathology images used for image feature extraction

Large whole-slide image files are excluded from Git tracking and must be obtained separately through an authorized data-access workflow.

## Proposed Methodology

The intended multimodal pipeline consists of the following stages:

1. **Clinical data preprocessing:** Prepare patient-level clinical features for model input.
2. **Histopathology image feature extraction:** Use pretrained ResNet-50 to extract image representations.
3. **Multimodal feature fusion:** Combine image and clinical representations.
4. **Patient similarity graph construction:** Construct a patient graph using cosine similarity.
5. **Graph Attention Network:** Learn patient-level representations through graph attention.
6. **Model evaluation:** Evaluate predictive performance using suitable classification metrics.

The implementation status and limitations of these stages are documented in `docs/multimodal_model.md`.

## Repository Structure

```text
breast-cancer-project/
├── backend/
│   ├── app.py
│   ├── database/
│   └── gdc/
├── data/
│   ├── metadata/
│   ├── models/
│   └── images/             # Large slide dataset excluded from Git
├── docs/
│   └── multimodal_model.md
├── figures/
├── scripts/
│   ├── patient_graph_model.py
│   └── multimodal_features.py
├── requirements.txt
├── requirements-model.txt
└── README.md
```

## Backend API

The project uses FastAPI to provide patient and dataset information.

Documented endpoints:

| Endpoint | Description |
|---|---|
| `/patients` | Patient metadata |
| `/patient/{case_uuid}` | Information for an individual patient |
| `/stats/diagnosis` | Diagnosis statistics |
| `/stats/patients` | Patient statistics |
| `/stats/slides` | Histopathology slide statistics |

## Installation

Clone the repository:

```bash
git clone https://github.com/bhavyasree-bunny/Breast-Cancer.git
cd Breast-Cancer
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv-model
source .venv-model/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For model-specific dependencies:

```bash
pip install -r requirements-model.txt
```

Additional dataset preparation, model training, and execution instructions are provided in the project documentation.

## Model Implementation

The currently documented model entry point is:

`script​s/patient_graph_model.py`

Shared preprocessing and multimodal fusion functions are described in:

`script​s/multimodal_features.py`

For the implementation workflow, including preparation, pilot feature extraction, training, and limitations, see [Multimodal Model Implementation](docs/multimodal_model.md).

## Research Status

This repository is under active development.

The proposed architecture and available experimental components should not be interpreted as a fully validated clinical diagnostic system. Reported results should be assessed together with their dataset split, evaluation protocol, and limitations.

## Disclaimer

This project is intended for research and educational purposes only. It is not a clinically validated diagnostic tool and must not be used for medical decision-making.