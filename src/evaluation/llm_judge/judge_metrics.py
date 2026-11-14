def flatten_expected_paths(data, prefix=""):
    """Return leaf paths from a nested annotation."""

    paths = []

    if isinstance(data, dict):
        for key, value in data.items():
            if key in {"document_id", "document_type", "synthetic",
                       "ground_truth_provenance", "visual_template_id",
                       "original_document_id", "data_origin"}:
                continue

            current = f"{prefix}.{key}" if prefix else key
            paths.extend(flatten_expected_paths(value, current))

    elif isinstance(data, list):
        for index, value in enumerate(data):
            current = f"{prefix}[{index}]"
            paths.extend(flatten_expected_paths(value, current))

    else:
        if prefix:
            paths.append(prefix)

    return paths


def validate_judge_response(response, expected_paths):
    if not isinstance(response, dict):
        raise ValueError("Judge response must be a JSON object.")

    assessments = response.get("field_assessments")
    extras = response.get("extra_predicted_fields", [])

    if not isinstance(assessments, list):
        raise ValueError("field_assessments must be a list.")

    if not isinstance(extras, list):
        raise ValueError("extra_predicted_fields must be a list.")

    returned = {}

    for item in assessments:
        if not isinstance(item, dict):
            raise ValueError("Invalid field assessment.")

        path = item.get("field_path")
        status = item.get("status")

        if status not in {"correct", "incorrect", "missing"}:
            raise ValueError(f"Invalid status for {path}: {status}")

        if path in returned:
            raise ValueError(f"Duplicate field assessment: {path}")

        returned[path] = item

    expected_set = set(expected_paths)
    returned_set = set(returned)

    missing_paths = expected_set - returned_set
    unexpected_paths = returned_set - expected_set

    if missing_paths:
        raise ValueError(
            f"Judge omitted expected fields: {sorted(missing_paths)}"
        )

    if unexpected_paths:
        raise ValueError(
            f"Judge returned unexpected field paths: {sorted(unexpected_paths)}"
        )

    completeness = response.get("semantic_completeness")
    understanding = response.get("contextual_understanding")

    if not isinstance(completeness, (int, float)):
        raise ValueError("semantic_completeness must be numeric.")

    if not 0 <= completeness <= 100:
        raise ValueError("semantic_completeness must be between 0 and 100.")

    if not isinstance(understanding, (int, float)):
        raise ValueError("contextual_understanding must be numeric.")

    if not 0 <= understanding <= 5:
        raise ValueError("contextual_understanding must be between 0 and 5.")

    if not all(isinstance(x, dict) and x.get("field_path") for x in extras):
        raise ValueError("Every extra field must contain a field_path.")

    return returned, extras


def calculate_invoice_metrics(response, expected_paths):
    assessments, extras = validate_judge_response(
        response, expected_paths
    )

    correct = sum(
        item["status"] == "correct"
        for item in assessments.values()
    )

    expected_count = len(expected_paths)
    extra_count = len(extras)

    denominator = expected_count + extra_count

    semantic_field_accuracy = (
        (correct / denominator) * 100
        if denominator else 0.0
    )

    return {
        "semantic_field_accuracy": round(
            semantic_field_accuracy, 2
        ),
        "semantic_completeness": round(
            float(response["semantic_completeness"]), 2
        ),
        "contextual_understanding": round(
            float(response["contextual_understanding"]), 2
        ),
        "expected_fields": expected_count,
        "correct_fields": correct,
        "incorrect_fields": sum(
            item["status"] == "incorrect"
            for item in assessments.values()
        ),
        "missing_fields": sum(
            item["status"] == "missing"
            for item in assessments.values()
        ),
        "extra_predicted_fields": extra_count,
        "field_assessments": list(assessments.values()),
        "extra_field_details": extras,
        "overall_reasoning": response.get(
            "overall_reasoning", ""
        ),
    }


def aggregate_metrics(results):
    if not results:
        raise ValueError("No successful judge results to aggregate.")

    count = len(results)

    return {
        "evaluated_invoices": count,
        "semantic_field_accuracy": round(
            sum(r["semantic_field_accuracy"] for r in results) / count, 2
        ),
        "semantic_completeness": round(
            sum(r["semantic_completeness"] for r in results) / count, 2
        ),
        "contextual_understanding": round(
            sum(r["contextual_understanding"] for r in results) / count, 2
        ),
        "aggregation": "Macro average across invoices",
    }