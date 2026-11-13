"""Audit clinical inputs and tune k using validation loss; never load test images."""
import argparse
import copy
import json
from collections import Counter
from contextlib import redirect_stdout

from legacy_multimodal_gat import WORK, read, train, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    destination = WORK / "tuning"
    destination.mkdir(exist_ok=True)
    rows = [r for r in read(WORK / "patients.csv") if r["split"] != "test"
            and (WORK / "embeddings" / (r["case_uuid"] + ".npz")).exists()]
    audit = {"scope": "available training and validation patients only", "patients": len(rows), "features": {}}
    for field in ("age_at_diagnosis", "race", "ethnicity"):
        audit["features"][field] = {}
        for split in ("train", "validation"):
            subset = [r for r in rows if r["split"] == split]
            counts = Counter(r[field] for r in subset)
            missing = sum(n for v, n in counts.items() if v.strip().lower() in {"", "not reported", "unknown"})
            result = {"patients": len(subset), "missing": missing, "unique_nonmissing": sum(v.strip().lower() not in {"", "not reported", "unknown"} for v in counts)}
            if field != "age_at_diagnosis":
                result["counts"] = dict(counts)
            else:
                ages = [float(r[field])/365.25 for r in subset if r[field]]
                result["range_years"] = [min(ages), max(ages)] if ages else []
            audit["features"][field][split] = result
    audit["excluded_features"] = "Stage and gender are entirely missing in saved exports; clinical extraction is still a debug script."
    (destination / "clinical_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    summary = []
    for neighbours in (3, 5, 10):
        run = destination / f"k{neighbours}"
        run.mkdir(exist_ok=True)
        config = copy.copy(args)
        config.work, config.run_dir = WORK, run
        config.neighbours, config.validation_only = neighbours, True
        with (run / "run.log").open("w") as log, redirect_stdout(log):
            train(config)
        report = json.loads((run / "metrics.json").read_text())
        validation = report["evaluation"]["validation"]
        summary.append({"neighbours": neighbours, "validation_loss": report["best_validation_loss"],
                        "balanced_accuracy": validation["balanced_accuracy"], "roc_auc": validation["roc_auc"],
                        "best_epoch": report["best_epoch"]})
        print(summary[-1], flush=True)
    best = min(summary, key=lambda r: r["validation_loss"])
    write(destination / "summary.csv", summary)
    (destination / "selection.json").write_text(json.dumps({"criterion": "minimum validation cross-entropy", "selected": best,
        "test_evaluated": False, "seed": 42, "note": "Single-seed small search; original model preserved. Test has been viewed previously and is not a fresh confirmatory holdout."}, indent=2) + "\n")
    print("Selected", best)


if __name__ == "__main__":
    main()
