"""Create a reproducible, patient-representative sample of 500 SVS files."""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "metadata"
SOURCE = METADATA / "patient_slide_mapping.csv"
DESTINATION = METADATA / "selected_500_slides.csv"
SEED = 20260928


def main():
    with SOURCE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "patient_id", "case_uuid", "slide_file_uuid", "slide_file_name",
        "slide_count_for_patient",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Mapping is empty or missing required columns")
    if any(not row["slide_file_uuid"] for row in rows):
        raise ValueError("Mapping contains patients without a slide")
    if len({row["slide_file_uuid"] for row in rows}) != len(rows):
        raise ValueError("Expected one mapping row per distinct slide UUID")

    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row["patient_id"]].append(row)
    if len(by_patient) != 250:
        raise ValueError("Expected 250 patients, found %d" % len(by_patient))
    if any(not slides for slides in by_patient.values()):
        raise ValueError("Every patient must have at least one slide")

    rng = random.Random(SEED)
    # Sort inputs before seeded choices so output stays stable if the source row
    # order changes.
    patients = sorted(by_patient)
    selected = []
    used_uuids = set()
    for patient_id in patients:
        options = sorted(by_patient[patient_id], key=lambda r: r["slide_file_uuid"])
        row = dict(rng.choice(options))
        row["selection_group"] = "primary_slide"
        selected.append(row)
        used_uuids.add(row["slide_file_uuid"])

    additional_pool = [
        row for patient_id in patients if len(by_patient[patient_id]) > 1
        for row in by_patient[patient_id]
        if row["slide_file_uuid"] not in used_uuids
    ]
    if len(additional_pool) < 250:
        raise ValueError("Only %d additional distinct slides available" % len(additional_pool))
    additional = rng.sample(additional_pool, 250)
    for source_row in additional:
        row = dict(source_row)
        row["selection_group"] = "additional_slide"
        selected.append(row)

    fieldnames = [
        "patient_id", "case_uuid", "slide_file_uuid", "slide_file_name",
        "slide_count_for_patient", "selection_group",
    ]
    with DESTINATION.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    # Validate the written CSV, rather than only the in-memory sample.
    with DESTINATION.open(newline="") as handle:
        saved = list(csv.DictReader(handle))
    slide_uuids = [row["slide_file_uuid"] for row in saved]
    patient_counts = Counter(row["patient_id"] for row in saved)
    duplicate_count = len(slide_uuids) - len(set(slide_uuids))
    assert len(saved) == 500
    assert len(set(slide_uuids)) == 500
    assert len(patient_counts) == 250
    assert all(count >= 1 for count in patient_counts.values())
    assert sum(row["selection_group"] == "primary_slide" for row in saved) == 250
    assert sum(row["selection_group"] == "additional_slide" for row in saved) == 250
    assert duplicate_count == 0

    print("seed=%d" % SEED)
    print("total_selected_slides=%d" % len(saved))
    print("unique_patients_represented=%d" % len(patient_counts))
    print("patients_with_1_selected_slide=%d" % sum(n == 1 for n in patient_counts.values()))
    print("patients_with_2plus_selected_slides=%d" % sum(n >= 2 for n in patient_counts.values()))
    print("duplicate_slide_uuid_count=%d" % duplicate_count)
    print("first_10_rows:")
    print(",".join(fieldnames))
    for row in saved[:10]:
        print(",".join(row[field] for field in fieldnames))


if __name__ == "__main__":
    main()
