import json
import time

from openai import OpenAI

from .judge_config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
)

from .judge_prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)

from .judge_metrics import calculate_invoice_metrics


if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Check your project-root .env file."
    )


client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=REQUEST_TIMEOUT,
)


def judge_one_invoice(
    document_id,
    ocr_text,
    ground_truth,
    prediction,
    expected_paths,
    ocr_was_truncated=False,
):
    user_prompt = build_user_prompt(
        document_id=document_id,
        ocr_text=ocr_text,
        ground_truth=ground_truth,
        prediction=prediction,
        expected_field_paths=expected_paths,
        ocr_was_truncated=ocr_was_truncated,
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            raw_response = response.choices[0].message.content
            parsed_response = json.loads(raw_response)

            metrics = calculate_invoice_metrics(
                parsed_response,
                expected_paths,
            )

            return {
                "document_id": document_id,
                "judge_model": OPENAI_MODEL,
                "judge_status": "success",
                **metrics,
                "raw_judge_response": parsed_response,
                "token_usage": (
                    response.usage.total_tokens
                    if response.usage else None
                ),
            }

        except Exception as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Judge failed for {document_id} after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )