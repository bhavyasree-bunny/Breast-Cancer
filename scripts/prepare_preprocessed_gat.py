"""Build patient embeddings and a frozen split from cohort-preprocessed patches."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path(__file__).resolve().parents[1]
PATCH_MANIFEST = ROOT / "data/processed_patches/cohort_v1/model_patches.csv"
PATIENTS_CSV = ROOT / "data/metadata/patients_250.csv"
CLINICAL_CSV = ROOT / "data/metadata/clinical.csv"
DEFAULT_WORK = ROOT / "data/models/preprocessed_patient_graph/work"
TARGETS = {"Infiltrating duct carcinoma, NOS": 0, "Lobular carcinoma, NOS": 1}


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-manifest", type=Path, default=PATCH_MANIFEST)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.work.exists() and any(args.work.iterdir()):
        parser.error("work directory is not empty; choose a new --work to preserve frozen splits")
    args.work.mkdir(parents=True, exist_ok=True)
    patients = {r["case_uuid"]: r for r in read_csv(PATIENTS_CSV)}
    clinical = {r["case_uuid"]: r for r in read_csv(CLINICAL_CSV)}
    patch_rows = read_csv(args.patch_manifest)
    grouped = {}
    for row in patch_rows:
        p = patients.get(row["case_uuid"])
        if p is None or p["case_id"] != row["patient_id"]:
            raise ValueError("Patch manifest has an invalid patient association")
        if p["primary_diagnosis"] in TARGETS:
            grouped.setdefault(row["case_uuid"], []).append(row)
    cases = sorted(grouped)
    if not cases:
        raise ValueError("No patients with exact ductal/lobular target diagnoses")
    labels = np.asarray([TARGETS[patients[c]["primary_diagnosis"]] for c in cases])
    train, held = train_test_split(np.arange(len(cases)), test_size=.4,
                                   stratify=labels, random_state=args.seed)
    validation, test = train_test_split(held, test_size=.5,
                                       stratify=labels[held], random_state=args.seed)
    split_by_case = {cases[i]: split for split, indices in
                     (("train", train), ("validation", validation), ("test", test))
                     for i in indices}
    patient_rows = []
    for case in cases:
        p, c = patients[case], clinical.get(case)
        if c is None or c["case_id"] != p["case_id"]:
            raise ValueError("Missing or mismatched clinical row for " + case)
        patient_rows.append({"case_uuid": case, "patient_id": p["case_id"],
            "label": TARGETS[p["primary_diagnosis"]], "split": split_by_case[case],
            "age_at_diagnosis": c["age_at_diagnosis"], "race": c["race"], "ethnicity": c["ethnicity"]})
    write_csv(args.work / "patients.csv", patient_rows,
              ["case_uuid", "patient_id", "label", "split", "age_at_diagnosis", "race", "ethnicity"])

    torch.set_num_threads(4)
    cache = ROOT / "data/models/multimodal_gat/torch_cache"
    torch.hub.set_dir(str(cache))
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = weights.transforms()
    (args.work / "embeddings").mkdir()
    manifest_rows = []
    with torch.inference_mode():
        for p in patient_rows:
            rows = sorted(grouped[p["case_uuid"]], key=lambda r: r["patch_file"])
            vectors = []
            for row in rows:
                path = ROOT / row["patch_file"]
                with Image.open(path) as image:
                    image.load()
                    if image.mode != "RGB" or image.size != (224, 224):
                        raise ValueError("Invalid patch image: " + str(path))
                    tensor = transform(image)
                vectors.append(model(tensor.unsqueeze(0)).squeeze(0).numpy())
                manifest_rows.append({"case_uuid": p["case_uuid"], "patient_id": p["patient_id"],
                    "slide_file_uuid": row["slide_file_uuid"], "patch_file": row["patch_file"],
                    "x": row["x"], "y": row["y"], "level": row["level"], "mpp_x": row["mpp_x"]})
            feature = np.mean(vectors, axis=0).astype(np.float32)
            if feature.shape != (2048,) or not np.isfinite(feature).all():
                raise ValueError("Invalid patient embedding: " + p["case_uuid"])
            np.savez(args.work / "embeddings" / (p["case_uuid"] + ".npz"),
                     feature=feature, case_uuid=p["case_uuid"],
                     image_encoder="ResNet50_IMAGENET1K_V2", patches=len(vectors))
            print(p["patient_id"], len(vectors), "patches", flush=True)
    write_csv(args.work / "patch_manifest.csv", manifest_rows,
              ["case_uuid", "patient_id", "slide_file_uuid", "patch_file", "x", "y", "level", "mpp_x"])
    print("Patients:", len(patient_rows), "class counts:",
          {"ductal": int(np.sum(labels == 0)), "lobular": int(np.sum(labels == 1))})
    print("Split counts:", {s: sum(r["split"] == s for r in patient_rows)
          for s in ("train", "validation", "test")})
    print("Frozen work directory:", args.work)


if __name__ == "__main__":
    main()
