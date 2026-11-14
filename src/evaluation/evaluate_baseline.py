"""
StructuraAI / Lexsi Labs — deterministic invoice evaluator V2.
Evaluates saved predictions only; does not run inference or modify source files.
"""

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from auditkit import FieldF1, Sample


# Defaults for the current project. These can also be overridden with CLI args.
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = ROOT / "results/baseline_aya_expanse_8b/run_20261111_134303/baseline_predictions.jsonl"
DEFAULT_ANNOTATIONS = ROOT / "Dataset/Invoice_Dataset/annotations"
DEFAULT_OUTPUT = ROOT / "results/auditkit/baseline"

MODEL = "CohereLabs/aya-expanse-8b"
EXPERIMENT = "Experiment A - Schema-free"
RUN_ID = "20261111_134303"
EXPECTED_COUNT = 42

SUPPLIER_FIELDS = ("name", "address", "phone", "email", "website")
CUSTOMER_FIELDS = ("name", "address")
DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%d/%m/%Y", "%m/%d/%Y")
CURRENCY_PREFIXES = (
    "NZD", "AUD", "USD", "EUR", "GBP", "INR", "CAD", "SGD",
    "NZ$", "A$", "US$", "C$", "S$", "₹", "$", "€", "£", "¥",
)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_no} is not an object.")
                rows.append(row)
    return rows


def read_annotations(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {folder}")
    result = {}
    for path in sorted(folder.glob("*.json")):
        if path.name == "dataset_statistics.json":
            continue
        with path.open(encoding="utf-8") as f:
            row = json.load(f)
        if isinstance(row, dict) and row.get("document_id"):
            doc_id = str(row["document_id"])
            if doc_id in result:
                raise ValueError(f"Duplicate annotation ID: {doc_id}")
            result[doc_id] = row
    return result


def read_test_ids(path):
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Test-ID file not found: {path}")
    ids = {
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not ids:
        raise ValueError("Official test-ID file is empty.")
    return ids


def pick(obj, *keys):
    if not isinstance(obj, dict):
        return None
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def keep_fields(obj, allowed):
    if not isinstance(obj, dict):
        return {}
    return {k: obj[k] for k in allowed if k in obj and obj[k] is not None}


def canonical_target(gt):
    details = {
        "invoiceNumber": gt.get("invoice_number"),
        "date": gt.get("invoice_date"),
        "dueDate": gt.get("due_date"),
        "paymentTerms": gt.get("payment_terms"),
        "purchaseOrder": gt.get("purchase_order_number"),
    }
    details = {k: v for k, v in details.items() if v is not None}

    items = []
    raw_items = gt.get("line_items") or []
    if not isinstance(raw_items, list):
        raise ValueError("Ground-truth line_items is not a list.")
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("Ground-truth line item is not an object.")
        mapped = {
            "description": item.get("description"),
            "quantity": item.get("quantity"),
            "unitPrice": item.get("unit_price"),
            "tax": item.get("tax_rate"),
            "lineSubtotal": item.get("line_subtotal"),
            "lineTax": item.get("line_tax"),
            "amount": item.get("line_total"),
        }
        items.append({k: v for k, v in mapped.items() if v is not None})

    invoice = {
        "supplier": keep_fields(gt.get("supplier"), SUPPLIER_FIELDS),
        "customer": keep_fields(gt.get("customer"), CUSTOMER_FIELDS),
        "invoiceDetails": details,
        "lineItems": items,
        "currency": gt.get("currency"),
        "subtotal": gt.get("subtotal"),
        "tax": gt.get("tax_total"),
        "total": gt.get("total"),
        "paymentReference": gt.get("payment_reference"),
        "notes": gt.get("notes"),
    }
    invoice = {k: v for k, v in invoice.items()
               if v is not None and v != {} and v != []}
    return {"invoice": invoice}


def canonical_prediction(parsed):
    root = parsed.get("invoice", parsed)
    if not isinstance(root, dict):
        return {"invoice": {"_invalid_structure": root}}

    supplier = keep_fields(root.get("supplier"), SUPPLIER_FIELDS)
    customer = keep_fields(root.get("customer"), CUSTOMER_FIELDS)
    details_raw = root.get("invoiceDetails") or root.get("invoice_details") or {}
    if not isinstance(details_raw, dict):
        details_raw = {"_invalid_details_structure": details_raw}

    details = {
        "invoiceNumber": pick(details_raw, "invoiceNumber", "invoice_number", "number"),
        "date": pick(details_raw, "date", "invoiceDate", "invoice_date"),
        "dueDate": pick(details_raw, "dueDate", "due_date"),
        "paymentTerms": pick(details_raw, "paymentTerms", "payment_terms"),
        "purchaseOrder": pick(details_raw, "purchaseOrder", "purchaseOrderNumber",
                               "purchase_order_number"),
    }
    details = {k: v for k, v in details.items() if v is not None}

    raw_items = root.get("lineItems", root.get("line_items", [])) or []
    items = []
    if not isinstance(raw_items, list):
        items = [{"_invalid_line_items_structure": raw_items}]
    else:
        for item in raw_items:
            if not isinstance(item, dict):
                items.append({"_unmapped_item": item})
                continue
            mapped = {
                "description": pick(item, "description"),
                "quantity": pick(item, "quantity"),
                "unitPrice": pick(item, "unitPrice", "unit_price"),
                "tax": pick(item, "tax", "taxRate", "tax_rate"),
                "lineSubtotal": pick(item, "lineSubtotal", "line_subtotal"),
                "lineTax": pick(item, "lineTax", "line_tax"),
                "amount": pick(item, "amount", "lineTotal", "line_total", "total"),
            }
            items.append({k: v for k, v in mapped.items() if v is not None})

    invoice = {
        "supplier": supplier,
        "customer": customer,
        "invoiceDetails": details,
        "lineItems": items,
        "currency": pick(root, "currency"),
        "subtotal": pick(root, "subtotal"),
        "tax": pick(root, "tax", "taxTotal", "tax_total"),
        "total": pick(root, "total", "grandTotal", "grand_total"),
        "paymentReference": pick(root, "paymentReference", "payment_reference"),
        "notes": pick(root, "notes"),
    }
    invoice = {k: v for k, v in invoice.items()
               if v is not None and v != {} and v != []}

    known = {
        "invoice", "supplier", "customer", "invoiceDetails", "invoice_details",
        "lineItems", "line_items", "currency", "subtotal", "tax", "taxTotal",
        "tax_total", "total", "grandTotal", "grand_total",
        "paymentReference", "payment_reference", "notes",
    }
    extras = {k: v for k, v in root.items() if k not in known}
    if extras:
        invoice["_unmapped"] = extras
    return {"invoice": invoice}


def normalize_number(value):
    """Normalize strict numeric strings; never extract a number from arbitrary prose."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)

    original = str(value).strip()
    text = original.replace(",", "").strip()
    negative_parentheses = text.startswith("(") and text.endswith(")")
    if negative_parentheses:
        text = text[1:-1].strip()

    upper = text.upper()
    for prefix in sorted(CURRENCY_PREFIXES, key=len, reverse=True):
        if upper.startswith(prefix.upper()):
            text = text[len(prefix):].strip()
            break
    if text.endswith("%"):
        text = text[:-1].strip()

    if not NUMBER_RE.fullmatch(text):
        return original
    number = float(text)
    return -abs(number) if negative_parentheses else number


def count_leaves(value):
    if isinstance(value, dict):
        return sum(count_leaves(v) for v in value.values())
    if isinstance(value, list):
        return sum(count_leaves(v) for v in value)
    return 1


def scalar_equal(a, b):
    a, b = normalize_number(a), normalize_number(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9
    return str(a).strip().casefold() == str(b).strip().casefold()


def equal_shared_leaves(target, pred):
    if isinstance(target, dict):
        if not isinstance(pred, dict):
            return 0
        return sum(equal_shared_leaves(v, pred[k]) for k, v in target.items() if k in pred)
    if isinstance(target, list):
        if not isinstance(pred, list):
            return 0
        return sum(equal_shared_leaves(target[i], pred[j])
                   for i, j in align_items(target, pred))
    return int(scalar_equal(target, pred))


def align_items(target_items, pred_items):
    """Greedy one-to-one item pairing by equal shared scalar leaves."""
    candidates = []
    for ti, target in enumerate(target_items):
        for pi, pred in enumerate(pred_items):
            candidates.append((equal_shared_leaves(target, pred), ti, pi))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_t, used_p, pairs = set(), set(), []
    for score, ti, pi in candidates:
        if ti not in used_t and pi not in used_p:
            used_t.add(ti)
            used_p.add(pi)
            pairs.append((ti, pi))
    return pairs


def coverage_count(target, pred):
    """Count expected leaves represented by a prediction path, even if wrong."""
    if isinstance(target, dict):
        if not isinstance(pred, dict):
            return 0
        return sum(coverage_count(v, pred[k]) for k, v in target.items() if k in pred)
    if isinstance(target, list):
        if not isinstance(pred, list):
            return 0
        return sum(coverage_count(target[i], pred[j]) for i, j in align_items(target, pred))
    return 1


def make_metric():
    numeric_paths = (
        "invoice.lineItems[].quantity",
        "invoice.lineItems[].unitPrice",
        "invoice.lineItems[].tax",
        "invoice.lineItems[].lineSubtotal",
        "invoice.lineItems[].lineTax",
        "invoice.lineItems[].amount",
        "invoice.subtotal",
        "invoice.tax",
        "invoice.total",
    )
    return FieldF1(
        case_sensitive=False,
        strip=True,
        date_formats=DATE_FORMATS,
        normalizers={path: normalize_number for path in numeric_paths},
    )


def parse_saved_prediction(row):
    # The saved inference flag is authoritative. Do not reparse raw text to
    # calculate parse rate; that previously produced a conflicting result.
    flag = row.get("json_parseable")
    parsed = row.get("parsed_prediction")
    if flag is True and isinstance(parsed, dict):
        return True, parsed, None
    if flag is False:
        return False, None, str(row.get("parse_error") or "Marked invalid by inference.")
    if flag is True:
        return False, None, "Inconsistent record: parseable=True but parsed_prediction is not an object."
    return False, None, "Missing json_parseable flag."


def evaluate(predictions, annotations):
    metric = make_metric()
    per_invoice = []
    tp_total = pred_total = target_total = coverage_total = 0
    parseable_total = exact_total = 0

    for row in predictions:
        doc_id = str(row["document_id"])
        target = canonical_target(annotations[doc_id])
        valid, parsed, error = parse_saved_prediction(row)

        if valid:
            prediction = canonical_prediction(parsed)
            output = json.dumps(prediction, ensure_ascii=False)
        else:
            prediction = None
            output = str(row.get("raw_model_output") or "")

        sample = Sample(
            input=f"Saved invoice extraction prediction: {doc_id}",
            target=json.dumps(target, ensure_ascii=False),
            id=doc_id,
            task="invoice_field_extraction",
        )
        scores = {s.name: s for s in metric.score(sample, output)}
        f1_obj = scores["field_f1"]
        precision = float(scores["field_precision"].value)
        recall = float(scores["field_recall"].value)
        f1 = float(f1_obj.value)

        if valid:
            metadata = f1_obj.metadata or {}
            tp = int(metadata.get("tp", 0))
            n_pred = int(metadata.get("n_pred", 0))
            n_target = int(metadata.get("n_target", count_leaves(target)))
            covered = coverage_count(target, prediction)
        else:
            tp, n_pred, n_target, covered = 0, 0, count_leaves(target), 0

        exact = bool(valid and tp == n_target and n_pred == n_target)
        tp_total += tp
        pred_total += n_pred
        target_total += n_target
        coverage_total += covered
        parseable_total += int(valid)
        exact_total += int(exact)

        per_invoice.append({
            "document_id": doc_id,
            "json_parseable": valid,
            "parse_error": error,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_invoice_match": exact,
            "true_positive_fields": tp,
            "predicted_fields": n_pred,
            "expected_fields": n_target,
            "represented_expected_fields": covered,
            "field_coverage": covered / n_target if n_target else 0.0,
        })

    micro_p = tp_total / pred_total if pred_total else 0.0
    micro_r = tp_total / target_total if target_total else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    macro_f1 = statistics.mean(x["f1"] for x in per_invoice) if per_invoice else 0.0
    n = len(predictions)

    metrics = {
        "field_micro_f1": micro_f1,
        "field_macro_f1": macro_f1,
        "field_precision_micro": micro_p,
        "field_recall_micro": micro_r,
        "exact_invoice_match_rate": exact_total / n if n else 0.0,
        "json_parse_rate": parseable_total / n if n else 0.0,
        "field_coverage": coverage_total / target_total if target_total else 0.0,
    }
    counts = {
        "true_positive_fields": tp_total,
        "predicted_fields": pred_total,
        "expected_fields": target_total,
        "represented_expected_fields": coverage_total,
        "exact_invoice_matches": exact_total,
        "json_parseable_predictions": parseable_total,
    }
    return metrics, counts, per_invoice


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_reports(out_dir, result):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, counts = result["metrics"], result["counts"]
    per_invoice = result["per_invoice"]
    pct = lambda x: f"{x * 100:.2f}%"

    json_path = out_dir / "baseline_metrics.json"
    csv_path = out_dir / "baseline_metrics.csv"
    md_path = out_dir / "baseline_report.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = [
        ("field_micro_f1", metrics["field_micro_f1"], counts["true_positive_fields"], counts["predicted_fields"]),
        ("field_macro_f1", metrics["field_macro_f1"], "", len(per_invoice)),
        ("field_precision_micro", metrics["field_precision_micro"], counts["true_positive_fields"], counts["predicted_fields"]),
        ("field_recall_micro", metrics["field_recall_micro"], counts["true_positive_fields"], counts["expected_fields"]),
        ("exact_invoice_match_rate", metrics["exact_invoice_match_rate"], counts["exact_invoice_matches"], len(per_invoice)),
        ("json_parse_rate", metrics["json_parse_rate"], counts["json_parseable_predictions"], len(per_invoice)),
        ("field_coverage", metrics["field_coverage"], counts["represented_expected_fields"], counts["expected_fields"]),
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "numerator", "denominator"])
        writer.writerows(rows)

    lines = [
        "# Aya Expanse 8B — Baseline Evaluation",
        "",
        f"- Model: `{result['model']}`",
        f"- Experiment: {result['experiment']}",
        f"- Baseline run: `{result['baseline_run_id']}`",
        "- Evaluation: deterministic; saved predictions only; no inference rerun",
        f"- Test documents: {result['test_documents']}",
        "- Test set: all 42 invoices are synthetic",
        "- Dataset: 800 records; 280 complete field-level extraction pairs",
        "- Split: 196 train / 42 validation / 42 test",
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
        "## Methodology and limitations",
        "",
        "- AuditKIT FieldF1 is used for field precision, recall, and F1.",
        "- Micro F1 aggregates field counts; Macro F1 averages the per-invoice F1 scores.",
        "- AuditKIT performs order-independent matching for lists of line-item objects.",
        "- Field Coverage is structural representation, not correctness. Its line-item pairing uses a deterministic greedy match based on equal shared scalar fields.",
        "- Numeric normalization accepts strict numeric strings with recognized currency prefixes, commas, and percentage suffixes. It does not extract numbers from arbitrary text.",
        "- Dates use the configured formats; ambiguous numeric dates depend on the dataset's date convention.",
        "- `supplier.tax_id` and document/provenance metadata are excluded. Supplier/customer scoring uses explicit field whitelists.",
        "- Ground-truth line subtotals and taxes are not inferred from other fields.",
        "- Invalid JSON is not repaired and receives zero field scores.",
        "- Unsupported-value rate is not calculated without OCR evidence verification.",
        "- Latency and cost are excluded from model-quality metrics and should be evaluated separately for ROI.",
        "- These results apply only to this synthetic 42-document test set; do not present them as real-world performance.",
        "",
        "## Per-invoice results",
        "",
        "| Document | JSON valid | Precision | Recall | F1 | Coverage | Exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for x in per_invoice:
        lines.append(
            f"| {x['document_id']} | {'Yes' if x['json_parseable'] else 'No'} "
            f"| {pct(x['precision'])} | {pct(x['recall'])} | {pct(x['f1'])} "
            f"| {pct(x['field_coverage'])} | {'Yes' if x['exact_invoice_match'] else 'No'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved invoice predictions using AuditKIT.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-ids", type=Path, default=None,
                        help="Optional official test split IDs file, one ID per line.")
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT,
                        help="Set to 0 to disable the expected-count check.")
    args = parser.parse_args()

    predictions_path = args.predictions.resolve()
    annotations_dir = args.annotations.resolve()
    output_dir = args.output.resolve()

    predictions = read_jsonl(predictions_path)
    annotations = read_annotations(annotations_dir)
    if not predictions:
        raise ValueError("No prediction records found.")

    ids = [str(row.get("document_id", "")) for row in predictions]
    duplicates = [doc_id for doc_id, count in Counter(ids).items() if count > 1]
    blank_ids = [i for i, doc_id in enumerate(ids) if not doc_id]
    missing_annotations = [doc_id for doc_id in ids if doc_id not in annotations]
    if duplicates or blank_ids or missing_annotations:
        raise ValueError(
            f"Alignment failure. Duplicates={duplicates}; blank ID rows={blank_ids}; "
            f"missing annotations={missing_annotations}"
        )

    if args.expected_count and len(predictions) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} predictions; found {len(predictions)}.")

    official_ids = read_test_ids(args.test_ids)
    split_verified = False
    if official_ids is not None:
        predicted_ids = set(ids)
        missing = sorted(official_ids - predicted_ids)
        extra = sorted(predicted_ids - official_ids)
        if missing or extra:
            raise ValueError(f"Official test split mismatch. Missing={missing}; extra={extra}")
        split_verified = True

    metrics, counts, per_invoice = evaluate(predictions, annotations)
    try:
        auditkit_version = version("auditkit")
    except PackageNotFoundError:
        auditkit_version = "unknown"

    result = {
        "model": MODEL,
        "experiment": EXPERIMENT,
        "baseline_run_id": RUN_ID,
        "evaluation_type": "deterministic_saved_prediction_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "auditkit_version": auditkit_version,
        "test_documents": len(predictions),
        "official_test_split_verified": split_verified,
        "test_id_verification_note": (
            "Exact match against supplied official test-ID file."
            if split_verified else
            "Not checked against an official test-ID file; annotation existence and unique IDs were checked."
        ),
        "input_files": {
            "predictions_path": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "annotations_directory": str(annotations_dir),
        },
        "dataset": {
            "total_records": 800,
            "complete_field_level_pairs": 280,
            "train": 196,
            "validation": 42,
            "test": 42,
            "records_without_complete_extraction_labels": 520,
            "test_provenance": "All 42 test invoices are synthetic.",
        },
        "metrics": metrics,
        "counts": counts,
        "policy": {
            "primary_metrics": ["field_micro_f1", "field_macro_f1"],
            "supporting_metrics": [
                "field_precision_micro", "field_recall_micro",
                "exact_invoice_match_rate", "json_parse_rate", "field_coverage",
            ],
            "tax_id": "Excluded based on prior OCR visibility review.",
            "metadata": "Document/provenance metadata excluded from targets.",
            "invalid_json": "Not repaired; saved inference parse flag is used.",
            "unsupported_value_rate": "Not calculated; requires OCR evidence verification.",
            "latency_cost": "Excluded from model-quality metrics; analyze separately for ROI.",
            "financial_field_accuracy": "Not calculated as a separate metric.",
        },
        "per_invoice": per_invoice,
    }

    write_reports(output_dir, result)
    pct = lambda x: f"{x * 100:.2f}%"
    print("=" * 62)
    print("DETERMINISTIC BASELINE EVALUATION COMPLETED")
    print("=" * 62)
    print(f"Test documents:       {len(predictions)}")
    print(f"Official split check: {'PASSED' if split_verified else 'NOT PERFORMED'}")
    print(f"JSON Parse Rate:      {pct(metrics['json_parse_rate'])}")
    print(f"Field Micro F1:       {pct(metrics['field_micro_f1'])}")
    print(f"Field Macro F1:       {pct(metrics['field_macro_f1'])}")
    print(f"Micro Precision:      {pct(metrics['field_precision_micro'])}")
    print(f"Micro Recall:         {pct(metrics['field_recall_micro'])}")
    print(f"Exact Invoice Match:  {pct(metrics['exact_invoice_match_rate'])}")
    print(f"Field Coverage:       {pct(metrics['field_coverage'])}")
    print("\nReports saved:")
    print(output_dir / "baseline_metrics.json")
    print(output_dir / "baseline_metrics.csv")
    print(output_dir / "baseline_report.md")

if __name__ == "__main__":
    main()
