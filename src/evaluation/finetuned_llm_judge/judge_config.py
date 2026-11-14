import os
from pathlib import Path
from dotenv import load_dotenv


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# Load environment variables from project-root .env
load_dotenv(PROJECT_ROOT / ".env")


# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv(
    "OPENAI_JUDGE_MODEL",
    "gpt-4.1-mini"
)


# Dataset paths
DATASET_ROOT = (
    PROJECT_ROOT
    / "Dataset"
    / "Invoice_Dataset"
)

OCR_TEXT_DIR = DATASET_ROOT / "ocr_text"

ANNOTATIONS_DIR = DATASET_ROOT / "annotations"


# Fine-tuned Aya Expanse 8B predictions
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "finetuned_aya_expanse_8b"
    / "run_20261113_103651"
    / "finetuned_predictions.jsonl"
)


# Fine-tuned LLM Judge output directory
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "auditkit"
    / "finetuned"
    / "llm_judge"
)


# Output files
RESULTS_JSONL = OUTPUT_DIR / "judge_results.jsonl"

METRICS_JSON = OUTPUT_DIR / "judge_metrics.json"

METRICS_CSV = OUTPUT_DIR / "judge_metrics.csv"

REPORT_MD = OUTPUT_DIR / "judge_report.md"


# API settings
MAX_RETRIES = 1

REQUEST_TIMEOUT = 120


# Maximum OCR characters sent to the judge.
# Set to 0 to disable truncation.
MAX_OCR_CHARACTERS = 30000


# Exclude metadata fields
EXCLUDED_FIELDS = {
    "document_id",
    "document_type",
    "synthetic",
    "ground_truth_provenance",
    "visual_template_id",
    "original_document_id",
    "data_origin",
}


# Exclude fields previously identified as not visible in OCR.
EXCLUDED_PATHS = {
    "supplier.tax_id",
}