
import csv
import json
import hashlib
import statistics
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError

# Reuse the original baseline evaluator's scoring logic.
import evaluate_baseline as evaluator


# --------------------------------------------------
# PATHS AND EXPERIMENT METADATA
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    ROOT / "results/finetuned_aya_expanse_8b/"
    "run_20261113_103651/finetuned_predictions.jsonl"
)

BASELINE_PREDICTIONS_PATH = (
    ROOT / "results/baseline_aya_expanse_8b/"
    "run_20261111_134303/baseline_predictions.jsonl"
)

ANNOTATIONS_PATH = ROOT / "Dataset/Invoice_Dataset/annotations"

OUTPUT_DIR = ROOT / "results/auditkit/finetuned"

RUN_ID = "20261113_103651"
EXPERIMENT = "Experiment B - Fine-tuned Schema-free"
MODEL = "CohereLabs/aya-expanse-8b"
ADAPTER = "rakesh-bishnoi29/structuraai-aya-expanse-8b-finetuned-extractor"

EXPECTED_COUNT = 42


# --------------------------------------------------
# VALIDATE INPUT FILES
# --------------------------------------------------

def validate_inputs():

    for path in [
        PREDICTIONS_PATH,
        BASELINE_PREDICTIONS_PATH,
        ANNOTATIONS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    predictions = evaluator.read_jsonl(PREDICTIONS_PATH)
    baseline_predictions = evaluator.read_jsonl(
        BASELINE_PREDICTIONS_PATH
    )

    if len(predictions) != EXPECTED_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COUNT} fine-tuned predictions, "
            f"found {len(predictions)}"
        )

    if len(baseline_predictions) != EXPECTED_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COUNT} baseline predictions, "
            f"found {len(baseline_predictions)}"
        )

    finetuned_ids = [
        str(row.get("document_id", "")) for row in predictions
    ]

    baseline_ids = [
        str(row.get("document_id", "")) for row in baseline_predictions
    ]

    if len(set(finetuned_ids)) != len(finetuned_ids):
        raise ValueError("Duplicate document IDs in fine-tuned predictions.")

    if finetuned_ids != baseline_ids:
        raise ValueError(
            "Fine-tuned and baseline document IDs/order do not match."
        )

    print("Input validation passed.")
    print(f"Fine-tuned predictions: {len(predictions)}")
    print(f"Baseline predictions:   {len(baseline_predictions)}")
    print("Document IDs and order: MATCH")

    return predictions


# --------------------------------------------------
# RUN DETERMINISTIC EVALUATION
# --------------------------------------------------

def run_evaluation():

    predictions = validate_inputs()

    annotations = evaluator.read_annotations(ANNOTATIONS_PATH)

    prediction_ids = [
        str(row["document_id"]) for row in predictions
    ]

    missing = [
        doc_id for doc_id in prediction_ids
        if doc_id not in annotations
    ]

    if missing:
        raise ValueError(f"Missing ground-truth annotations: {missing}")

    metrics, counts, per_invoice = evaluator.evaluate(
        predictions,
        annotations
    )

    # AuditKIT version
    try:
        auditkit_version = version("auditkit")
    except PackageNotFoundError:
        auditkit_version = "unknown"

    result = {
        "model": MODEL,
        "adapter": ADAPTER,
        "experiment": EXPERIMENT,
        "finetuned_run_id": RUN_ID,
        "evaluation_type": "deterministic_saved_prediction_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditkit_version": auditkit_version,
        "test_documents": len(predictions),

        "input_files": {
            "predictions_path": str(PREDICTIONS_PATH),
            "predictions_sha256": evaluator.sha256_file(
                PREDICTIONS_PATH
            ),
            "annotations_directory": str(ANNOTATIONS_PATH),
        },

        "dataset": {
            "train": 196,
            "validation": 42,
            "test": 42,
            "test_provenance": "All 42 test invoices are synthetic."
        },

        "metrics": metrics,
        "counts": counts,
        "per_invoice": per_invoice,

        "policy": {
            "primary_metrics": [
                "field_micro_f1",
                "field_macro_f1"
            ],
            "supporting_metrics": [
                "field_precision_micro",
                "field_recall_micro",
                "exact_invoice_match_rate",
                "json_parse_rate",
                "field_coverage"
            ],
            "latency_cost": (
                "Excluded from model-quality metrics; "
                "analyze separately for ROI."
            )
        }
    }

    # --------------------------------------------------
    # SAVE JSON REPORT
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "finetuned_metrics.json"

    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # --------------------------------------------------
    # SAVE CSV REPORT
    # --------------------------------------------------

    csv_path = OUTPUT_DIR / "finetuned_metrics.csv"

    pct_rows = [
        ("field_micro_f1", metrics["field_micro_f1"]),
        ("field_macro_f1", metrics["field_macro_f1"]),
        ("field_precision_micro", metrics["field_precision_micro"]),
        ("field_recall_micro", metrics["field_recall_micro"]),
        ("exact_invoice_match_rate", metrics["exact_invoice_match_rate"]),
        ("json_parse_rate", metrics["json_parse_rate"]),
        ("field_coverage", metrics["field_coverage"]),
    ]

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(["metric", "value"])

        for name, value in pct_rows:
            writer.writerow([name, value])

    # --------------------------------------------------
    # SAVE MARKDOWN REPORT
    # --------------------------------------------------

    md_path = OUTPUT_DIR / "finetuned_report.md"

    pct = lambda value: f"{value * 100:.2f}%"

    lines = [
        "# Aya Expanse 8B — Fine-tuned Evaluation",
        "",
        f"- Model: `{MODEL}`",
        f"- Adapter: `{ADAPTER}`",
        f"- Experiment: {EXPERIMENT}",
        f"- Run ID: `{RUN_ID}`",
        "- Evaluation: deterministic; saved predictions only",
        f"- Test documents: {len(predictions)}",
        "- Test set: all 42 invoices are synthetic.",
        "",
        "## Results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| **Field-level Micro F1** | **{pct(metrics['field_micro_f1'])}** |",
        f"| **Field-level Macro F1** | **{pct(metrics['field_macro_f1'])}** |",
        f"| Field Precision (micro) | {pct(metrics['field_precision_micro'])} |",
        f"| Field Recall (micro) | {pct(metrics['field_recall_micro'])} |",
        f"| Exact Invoice Match Rate | {pct(metrics['exact_invoice_match_rate'])} |",
        f"| JSON Parse Rate | {pct(metrics['json_parse_rate'])} |",
        f"| Field Coverage | {pct(metrics['field_coverage'])} |",
        "",
        "## Methodology",
        "",
        "- Uses the same AuditKIT FieldF1 scoring implementation as baseline.",
        "- Uses the same ground-truth annotation directory.",
        "- Uses the same 42 test invoices as baseline.",
        "- Invalid JSON is not repaired.",
        "- Micro F1 aggregates field counts.",
        "- Macro F1 averages per-invoice F1 scores.",
        "- Latency and cost are evaluated separately.",
        "- All test invoices are synthetic; results should not be "
        "presented as real-world performance.",
        "",
        "## Per-invoice results",
        "",
        "| Document | JSON valid | Precision | Recall | F1 | Coverage | Exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in per_invoice:
        lines.append(
            f"| {row['document_id']} "
            f"| {'Yes' if row['json_parseable'] else 'No'} "
            f"| {pct(row['precision'])} "
            f"| {pct(row['recall'])} "
            f"| {pct(row['f1'])} "
            f"| {pct(row['field_coverage'])} "
            f"| {'Yes' if row['exact_invoice_match'] else 'No'} |"
        )

    md_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINE-TUNED DETERMINISTIC EVALUATION COMPLETED")
    print("=" * 60)

    print(f"Test documents:       {len(predictions)}")
    print(f"JSON Parse Rate:      {pct(metrics['json_parse_rate'])}")
    print(f"Field Micro F1:       {pct(metrics['field_micro_f1'])}")
    print(f"Field Macro F1:       {pct(metrics['field_macro_f1'])}")
    print(f"Micro Precision:      {pct(metrics['field_precision_micro'])}")
    print(f"Micro Recall:         {pct(metrics['field_recall_micro'])}")
    print(f"Exact Invoice Match:  {pct(metrics['exact_invoice_match_rate'])}")
    print(f"Field Coverage:       {pct(metrics['field_coverage'])}")

    print("\nReports saved:")

    print(json_path)
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    run_evaluation()
