# StructuraAI – Structured Information Extraction Using Fine-Tuned LLMs

**Building and fine-tuning small open-source LLMs to convert unstructured document data into structured, machine-readable JSON.**

## 🚀 Project Overview

StructuraAI explores how domain-specific fine-tuning can improve open-source Large Language Models (LLMs) for structured information extraction.

For this experiment, we focused on **Invoice Extraction** – converting unstructured invoice OCR text into structured JSON containing invoice details, supplier and customer information, line items, taxes, and financial information.

Why is structured information extraction important?

Whether we want to store information in databases, automate business workflows, generate reports, or integrate documents with other software systems, extracting information into a consistent and structured format is essential.

Our objective was to fine-tune a general-purpose open-source LLM to perform this task more accurately, completely, and reliably.

**Our experiment:** Fine-tuning [Aya Expanse 8B](https://huggingface.co/CohereLabs/aya-expanse-8b) using QLoRA and comparing its performance against the original base model.

## 🧠 Fine-Tuning Approach

We used **Quantized Low-Rank Adaptation (QLoRA)** to fine-tune Aya Expanse 8B for invoice information extraction.

| Configuration | Value |
|---|---|
| Base Model | Aya Expanse 8B |
| Fine-Tuning Method | QLoRA |
| Quantization | 4-bit NF4 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Epochs | 3 |
| Learning Rate | 5e-5 |
| Training Steps | 75 |

The resulting model is a LoRA adapter that works alongside the original Aya Expanse 8B base model.

## 📊 Before vs After Fine-Tuning

📑 **[Before-vs-After Report (PDF)](Submission/Before-vs-after.pdf)**

We evaluated both models using the same **test invoice examples**, maintaining the same extraction instructions, JSON schema, generation settings, and evaluation pipeline.

### 1. Deterministic Evaluation

| Metric | Base Aya 8B | Fine-Tuned Aya 8B | Improvement |
|---|---:|---:|---:|
| Field Micro F1 | 50.66% | **87.27%** | +36.61 pp |
| Field Macro F1 | 41.86% | **86.73%** | +44.87 pp |
| Precision | 65.98% | **87.08%** | +21.10 pp |
| Recall | 41.11% | **87.45%** | +46.34 pp |
| JSON Parse Rate | 97.62% | **100.00%** | +2.38 pp |
| Field Coverage | 41.64% | **89.22%** | +47.58 pp |
| Exact Invoice Match | 0% | 0% | No change |

### 2. LLM-as-a-Judge Evaluation

We also evaluated semantic correctness and contextual understanding using the same three-metric judge rubric for both models.

| Metric | Base Aya 8B | Fine-Tuned Aya 8B | Improvement |
|---|---:|---:|---:|
| Semantic Field Accuracy | 77.92% | **98.98%** | +21.06 pp |
| Semantic Completeness | 79.88% | **99.29%** | +19.41 pp |
| Contextual Understanding | 3.95/5 | **4.88/5** | +0.93 |

### 3. Inference Performance

| Metric | Base Aya 8B | Fine-Tuned Aya 8B |
|---|---:|---:|
| Average Latency / Invoice | 31.50 sec | 58.07 sec |
| Total Inference Time | 1,323.11 sec | 2,438.71 sec |


**Key Finding:** Field Micro F1 improved from 50.66% to 87.27%, a gain of 36.61 percentage points. The fine-tuned model also achieved a 100% JSON parse rate in this experiment.

However, average inference latency increased by approximately 84.3%. This highlights the trade-off between extraction quality and inference speed.


## 🔗 Project Resources

| Resource | Link |
|---|---|
| Invoice Dataset | [StructuraAI Dataset](https://huggingface.co/datasets/rakesh-bishnoi29/structuraai-dataset) |
| Base Model | [CohereLabs/Aya Expanse 8B](https://huggingface.co/CohereLabs/aya-expanse-8b) |
| Fine-Tuned Model Adapter | [StructuraAI Fine-Tuned Adapter](https://huggingface.co/rakesh-bishnoi29/structuraai-aya-expanse-8b-finetuned-extractor) |
| Hackathon Submission | [Laxsi Labs Final Submission](Submission/Laxsi-Labs-Submission.pdf) |



## 🔮 Future Vision

The current StructuraAI experiment focuses exclusively on invoice extraction.

Our long-term vision is to expand the approach to additional document categories:

- Invoices
- Forms
- Reports
- Clinical Notes
- Support Tickets

The goal is to build specialized open-source models capable of converting different types of unstructured and semi-structured documents into consistent, structured outputs that can be stored, processed, and integrated into automated workflows.

---

**Built as part of the Laxsi Labs Hackathon.**

**By RAKESH BISHNOI**