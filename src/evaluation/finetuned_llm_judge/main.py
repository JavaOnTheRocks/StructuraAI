import json
import sys
from pathlib import Path


# Allow direct execution:
# python src\evaluation\finetuned_llm_judge\main.py

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluation.finetuned_llm_judge.judge_config import (
    OCR_TEXT_DIR,
    ANNOTATIONS_DIR,
    PREDICTIONS_PATH,
    OUTPUT_DIR,
    RESULTS_JSONL,
    METRICS_JSON,
    METRICS_CSV,
    REPORT_MD,
    MAX_OCR_CHARACTERS,
    EXCLUDED_PATHS,
)

from src.evaluation.finetuned_llm_judge.judge_metrics import (
    flatten_expected_paths,
)

from src.evaluation.finetuned_llm_judge.judge_runner import (
    judge_one_invoice,
)

from src.evaluation.finetuned_llm_judge.judge_report import (
    save_reports,
)


# --------------------------------------------------
# 1. File reading utilities
# --------------------------------------------------

def read_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def read_jsonl(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:

                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSONL at {path}, "
                    f"line {line_number}: {exc}"
                )

    return records


def get_document_id(record):

    for key in (
        "document_id",
        "invoice_id",
        "id"
    ):

        value = record.get(key)

        if value:
            return str(value).strip()

    return None


# --------------------------------------------------
# 2. Remove excluded metadata
# --------------------------------------------------

def remove_excluded_fields(value, prefix=""):

    """Copy nested data while excluding non-evaluation metadata."""

    if isinstance(value, dict):

        cleaned = {}

        for key, child in value.items():

            current_path = (
                f"{prefix}.{key}"
                if prefix
                else key
            )

            if key in {
                "document_id",
                "document_type",
                "synthetic",
                "ground_truth_provenance",
                "visual_template_id",
                "original_document_id",
                "data_origin",
            }:

                continue

            if current_path in EXCLUDED_PATHS:
                continue

            cleaned[key] = remove_excluded_fields(
                child,
                current_path
            )

        return cleaned

    if isinstance(value, list):

        return [
            remove_excluded_fields(item, prefix)
            for item in value
        ]

    return value


# --------------------------------------------------
# 3. Load fine-tuned predictions
# --------------------------------------------------

def load_predictions():

    if not PREDICTIONS_PATH.exists():

        raise FileNotFoundError(
            f"Fine-tuned prediction file not found: "
            f"{PREDICTIONS_PATH}"
        )

    records = read_jsonl(PREDICTIONS_PATH)

    predictions = {}

    for record in records:

        document_id = get_document_id(record)

        if not document_id:

            raise ValueError(
                "A prediction record has no "
                "document_id, invoice_id, or id."
            )

        if document_id in predictions:

            raise ValueError(
                f"Duplicate prediction ID: {document_id}"
            )

        predictions[document_id] = record

    return predictions


# --------------------------------------------------
# 4. Load ground-truth annotations
# --------------------------------------------------

def load_annotations():

    if not ANNOTATIONS_DIR.exists():

        raise FileNotFoundError(
            f"Annotation directory not found: "
            f"{ANNOTATIONS_DIR}"
        )

    annotations = {}

    for path in ANNOTATIONS_DIR.glob("*.json"):

        annotation = read_json(path)

        document_id = (
            annotation.get("document_id")
            or annotation.get("invoice_id")
            or path.stem
        )

        document_id = str(document_id).strip()

        if document_id in annotations:

            raise ValueError(
                f"Duplicate annotation ID: {document_id}"
            )

        annotations[document_id] = annotation

    return annotations


# --------------------------------------------------
# 5. Load original OCR text
# --------------------------------------------------

def load_ocr(document_id):

    path = OCR_TEXT_DIR / f"{document_id}.txt"

    if not path.exists():

        raise FileNotFoundError(
            f"OCR file missing for {document_id}: {path}"
        )

    # Preserve original OCR text.
    text = path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    if not text.strip():

        raise ValueError(
            f"OCR file is empty for {document_id}: {path}"
        )

    was_truncated = False

    if (
        MAX_OCR_CHARACTERS > 0
        and len(text) > MAX_OCR_CHARACTERS
    ):

        text = text[:MAX_OCR_CHARACTERS]

        was_truncated = True

    return text, was_truncated


# --------------------------------------------------
# 6. Extract model prediction
# --------------------------------------------------

def get_prediction(record):

    """
    Use the stored parsed prediction when valid.
    Do not repair invalid JSON.
    """

    if record.get("json_parseable") is True:

        parsed = record.get("parsed_prediction")

        if isinstance(parsed, (dict, list)):
            return parsed

        raise ValueError(
            "json_parseable is true but parsed_prediction "
            "is not a JSON object or list."
        )

    for key in (
        "parsed_prediction",
        "prediction",
        "model_prediction",
        "generated_json",
    ):

        value = record.get(key)

        if isinstance(value, (dict, list)):
            return value

    # Preserve original invalid output.
    raw_output = (
        record.get("raw_model_output")
        or record.get("generated_text")
        or record.get("output")
        or record.get("response")
    )

    if raw_output is not None:

        return {

            "unparseable_model_output": raw_output,

            "note": (
                "The original model output was not valid parsed JSON. "
                "It has not been repaired."
            ),
        }

    return {

        "unparseable_model_output": None,

        "note": (
            "No parsed prediction or raw model output was found."
        ),
    }


# --------------------------------------------------
# 7. Verify official test split
# --------------------------------------------------

def check_test_split(prediction_ids):

    test_path = (
        PROJECT_ROOT
        / "Dataset"
        / "Invoice_Dataset"
        / "test.jsonl"
    )

    if not test_path.exists():

        print(
            "Official test split check: NOT PERFORMED "
            "(test.jsonl missing)"
        )

        return

    test_records = read_jsonl(test_path)

    test_ids = {
        get_document_id(record)
        for record in test_records
        if get_document_id(record)
    }

    if not test_ids:

        print(
            "Official test split check: NOT PERFORMED "
            "(could not identify IDs in test.jsonl)"
        )

        return

    if test_ids == prediction_ids:

        print(
            f"Official test split check: PASSED "
            f"({len(test_ids)} matching invoice IDs)"
        )

    else:

        missing = sorted(
            test_ids - prediction_ids
        )

        unexpected = sorted(
            prediction_ids - test_ids
        )

        raise ValueError(
            "Prediction IDs do not match official test.jsonl.\n"
            f"Missing predictions: {missing[:10]}\n"
            f"Unexpected predictions: {unexpected[:10]}"
        )


# --------------------------------------------------
# 8. Resume completed evaluations
# --------------------------------------------------

def load_completed():

    completed = {}

    if not RESULTS_JSONL.exists():
        return completed

    for record in read_jsonl(RESULTS_JSONL):

        if (
            record.get("judge_status") == "success"
            and record.get("document_id")
        ):

            completed[record["document_id"]] = record

    return completed


def append_result(result):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULTS_JSONL,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                result,
                ensure_ascii=False
            ) + "\n"
        )


# --------------------------------------------------
# 9. Validate inputs
# --------------------------------------------------

def validate_inputs(predictions, annotations):

    prediction_ids = set(predictions)

    annotation_ids = set(annotations)

    if not prediction_ids:

        raise ValueError(
            "No predictions found."
        )

    missing_annotations = (
        prediction_ids - annotation_ids
    )

    missing_predictions = (
        annotation_ids - prediction_ids
    )

    if missing_annotations:

        raise ValueError(
            "Missing annotations for prediction IDs: "
            f"{sorted(missing_annotations)}"
        )

    if missing_predictions:

        print(
            f"Note: {len(missing_predictions)} annotations "
            "have no matching prediction. "
            "They will not be evaluated."
        )

    missing_ocr = [

        document_id

        for document_id in sorted(prediction_ids)

        if not (
            OCR_TEXT_DIR / f"{document_id}.txt"
        ).exists()
    ]

    if missing_ocr:

        raise ValueError(
            f"Missing OCR files for: {missing_ocr}"
        )

    return prediction_ids


# --------------------------------------------------
# 10. Main evaluation pipeline
# --------------------------------------------------

def main():

    print("\n" + "=" * 65)

    print(
        "STRUCTURAAI - FINE-TUNED OCR-AWARE LLM JUDGE"
    )

    print("=" * 65)

    print("\nLoading fine-tuned predictions...")

    predictions = load_predictions()

    print("Loading ground-truth annotations...")

    annotations = load_annotations()

    print("Validating invoice IDs and OCR files...")

    prediction_ids = validate_inputs(
        predictions,
        annotations
    )

    check_test_split(prediction_ids)

    print(
        f"\nFine-tuned predictions found: {len(predictions)}"
    )

    print(
        f"Annotations found: {len(annotations)}"
    )

    print(
        f"OCR directory: {OCR_TEXT_DIR}"
    )

    print("\nRunning preflight checks...")

    prepared = {}

    for document_id in sorted(prediction_ids):

        ocr_text, was_truncated = load_ocr(
            document_id
        )

        ground_truth = remove_excluded_fields(
            annotations[document_id]
        )

        expected_paths = flatten_expected_paths(
            ground_truth
        )

        if not expected_paths:

            raise ValueError(
                f"No evaluable ground-truth fields "
                f"for {document_id}"
            )

        prediction = get_prediction(
            predictions[document_id]
        )

        prepared[document_id] = {

            "ocr_text": ocr_text,

            "ocr_was_truncated": was_truncated,

            "ground_truth": ground_truth,

            "prediction": prediction,

            "expected_paths": expected_paths,
        }

    print(
        f"Preflight passed: {len(prepared)} invoices "
        "have matching OCR, annotation, and prediction records."
    )

    # --------------------------------------------------
    # Load previously completed fine-tuned results
    # --------------------------------------------------

    completed = load_completed()

    results_by_id = dict(completed)

    remaining_ids = [

        document_id

        for document_id in sorted(prepared)

        if document_id not in completed
    ]

    print(
        f"\nAlready evaluated: {len(completed)}"
    )

    print(
        f"Remaining to evaluate: {len(remaining_ids)}"
    )

    if not remaining_ids:

        print(
            "\nAll invoices already have successful judge results."
        )

    else:

        print(
            "\nStarting fine-tuned OpenAI judge evaluation...\n"
        )

    # --------------------------------------------------
    # Evaluate each invoice
    # --------------------------------------------------

    for index, document_id in enumerate(
        remaining_ids,
        start=1
    ):

        item = prepared[document_id]

        print(
            f"[{index}/{len(remaining_ids)}] "
            f"Evaluating {document_id}..."
        )

        result = judge_one_invoice(

            document_id=document_id,

            ocr_text=item["ocr_text"],

            ground_truth=item["ground_truth"],

            prediction=item["prediction"],

            expected_paths=item["expected_paths"],

            ocr_was_truncated=item["ocr_was_truncated"],
        )

        result["ocr_was_truncated"] = (
            item["ocr_was_truncated"]
        )

        append_result(result)

        results_by_id[document_id] = result

        print(
            f"  A - Semantic Field Accuracy: "
            f"{result['semantic_field_accuracy']}%"
        )

        print(
            f"  B - Semantic Completeness: "
            f"{result['semantic_completeness']}%"
        )

        print(
            f"  C - Contextual Understanding: "
            f"{result['contextual_understanding']}/5"
        )

    # --------------------------------------------------
    # Generate final reports
    # --------------------------------------------------

    final_results = [

        results_by_id[document_id]

        for document_id in sorted(prepared)

        if (
            document_id in results_by_id
            and results_by_id[document_id].get(
                "judge_status"
            ) == "success"
        )
    ]

    if len(final_results) != len(prepared):

        raise RuntimeError(
            "Not all invoices have successful judge results. "
            "Reports were not finalized."
        )

    overall = save_reports(final_results)

    # --------------------------------------------------
    # Final terminal output
    # --------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "FINE-TUNED LLM JUDGE EVALUATION COMPLETED"
    )

    print("=" * 65)

    print(
        f"\nEvaluated invoices: "
        f"{overall['evaluated_invoices']}"
    )

    print(
        f"\nA - Semantic Field Accuracy: "
        f"{overall['semantic_field_accuracy']}%"
    )

    print(
        f"B - Semantic Completeness: "
        f"{overall['semantic_completeness']}%"
    )

    print(
        f"C - Contextual Understanding: "
        f"{overall['contextual_understanding']}/5"
    )

    print("\nSaved reports:")

    print(
        f"JSONL: {RESULTS_JSONL}"
    )

    print(
        f"Metrics: {METRICS_JSON}"
    )

    print(
        f"CSV: {METRICS_CSV}"
    )

    print(
        f"Report: {REPORT_MD}"
    )


if __name__ == "__main__":

    main()