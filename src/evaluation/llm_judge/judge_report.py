import csv
import json

from .judge_config import (
    OUTPUT_DIR,
    METRICS_JSON,
    METRICS_CSV,
    REPORT_MD,
)

from .judge_metrics import aggregate_metrics


def save_reports(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    overall = aggregate_metrics(results)

    # Overall JSON metrics
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    # Invoice-level CSV
    csv_columns = [
        "document_id",
        "semantic_field_accuracy",
        "semantic_completeness",
        "contextual_understanding",
        "expected_fields",
        "correct_fields",
        "incorrect_fields",
        "missing_fields",
        "extra_predicted_fields",
    ]

    with open(
        METRICS_CSV, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

        for result in results:
            writer.writerow({
                key: result.get(key)
                for key in csv_columns
            })

    # Markdown report
    lines = [
        "# StructuraAI - LLM-as-a-Judge Evaluation",
        "",
        "## Evaluation configuration",
        "",
        f"- Judge model: {results[0]['judge_model']}",
        f"- Evaluated invoices: {overall['evaluated_invoices']}",
        "- OCR source: original dataset OCR text files",
        "- Aggregation: macro average across invoices",
        "- Test dataset: 42 invoices, previously identified as synthetic",
        "",
        "## Overall results",
        "",
        "| Metric | Score |",
        "|---|---:|",
        (
            "| Semantic Field Accuracy | "
            f"{overall['semantic_field_accuracy']}% |"
        ),
        (
            "| Semantic Completeness | "
            f"{overall['semantic_completeness']}% |"
        ),
        (
            "| Contextual Understanding | "
            f"{overall['contextual_understanding']} / 5 |"
        ),
        "",
        "## Invoice-level results",
        "",
        "| Invoice | Field Accuracy | Completeness | Understanding |",
        "|---|---:|---:|---:|",
    ]

    for result in results:
        lines.append(
            f"| {result['document_id']} "
            f"| {result['semantic_field_accuracy']}% "
            f"| {result['semantic_completeness']}% "
            f"| {result['contextual_understanding']} / 5 |"
        )

    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- These LLM-judge scores are macro averages across invoices.",
        "- Semantic Field Accuracy uses the project-defined denominator:",
        "  expected fields plus judge-identified extra predicted fields.",
        "- OCR text is supplied as evidence; no new OCR is performed.",
        "- The test invoices are synthetic, so results should not be",
        "  presented as proof of real-world production performance.",
        "- LLM-judge scores are model-based assessments, not human-verified",
        "  ground truth. A human-reviewed sample is recommended.",
        "",
    ])

    REPORT_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return overall