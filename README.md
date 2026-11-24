# Explainable AI-Generated Text Detection via Rationale Distillation and GRPO

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-PEFT%20%7C%20TRL-yellow.svg)](https://huggingface.co/)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle%20GPU%20T4%20x2-20BEFF.svg)](https://www.kaggle.com/)

## 1. Overview

Conventional AI-generated text detectors (such as fine-tuned `RoBERTa-large` encoders) achieve strong classification accuracy but operate as **"black-box" classifiers**, outputting scalar probabilities without linguistic justification for their predictions.

This project presents an **Explainable AI-Generated Text Detection** pipeline evaluated across **11 text domains** built upon the **READ** benchmark framework (unifying **MAGE** and **L2R** corpora). Our proposed model, **`Qwen2.5-7B-Instruct` (SFT + GRPO)**, is trained in two stages:
1. **Stage 1 — Supervised Fine-Tuning (SFT):** Distills forensic Chain-of-Thought (CoT) rationales from a teacher model (`Qwen2.5-7B-Instruct`) so the student learns to analyze stylistic markers, sentence rhythm, lexical repetition, and structured JSON formatting.
2. **Stage 2 — Group Relative Policy Optimization (GRPO):** Applies critic-free reinforcement learning with a dual reward function optimizing structural compliance and verdict accuracy.

---

## 2. Repository Structure

```text
explainable-ai-text-detection/
├── README.md                                         # Project overview & summary of experimental results
├── notebooks/                                        # End-to-end experimental notebooks (with execution logs)
│   ├── 01a_data_preparation_splits.ipynb             # Data unification & stratified 11-domain splitting
│   ├── 02_teacher_rationale_distillation.ipynb       # Teacher Chain-of-Thought rationale distillation
│   ├── 03a_baseline_roberta_training.ipynb           # Training & evaluation of the RoBERTa-large baseline
│   ├── 03b_student_sft_training.ipynb                # Stage 1: QLoRA Supervised Fine-Tuning (SFT)
│   ├── 03c_student_grpo_training.ipynb               # Stage 2: Group Relative Policy Optimization (GRPO)
│   ├── 03d_qwen_fast_eval.ipynb                      # Inference & rationale generation on the 11-domain test set
│   └── 04_evaluation_and_explainability_analysis.ipynb # Metric aggregation, CoT analysis & 300 DPI figures
├── results/                                          # Evaluation tables & figures exported from Notebook 04
│   ├── table1_overall_metrics.csv                    # Table 1: Overall classification & format error comparison
│   ├── table2_domain_breakdown.csv                  # Table 2: Per-domain performance across 11 domains
│   ├── table3_explainability_quality.csv            # Table 3: Quantitative CoT rationale quality metrics
│   ├── qualitative_rationale_examples.csv            # Representative forensic linguistic rationale examples
│   ├── fig1_domain_macro_f1_comparison.png           # Cross-domain Macro-F1 bar chart (300 DPI)
│   └── fig2_confusion_matrices.png                   # Confusion matrices for Baseline vs. Student (300 DPI)
├── report/                                           # Full thesis/project report (LaTeX source & compiled PDFs)
│   ├── main_vi.pdf                                   # Vietnamese report
│   ├── main_en.pdf                                   # English report
│   └── src/                                          # LaTeX source files (.tex) and figure assets
└── slides/                                           # Presentation slides
```

*(Note: Raw corpora, processed parquet splits, and model adapter checkpoints are excluded from Git tracking due to file size limits and will be provided separately upon request).*

---

## 3. Key Experimental Results

All models are evaluated on a balanced stratified test set spanning **11 domains ($N = 550$, $50$ samples per domain)**:

### 3.1. Overall Classification Performance
| Model | Accuracy (%) | Macro-F1 (%) | AI F1 (%) | Human F1 (%) | Format Error Rate (%) | Explainability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: RoBERTa-large** | **90.91** | **90.89** | **91.32** | **90.46** | **0.00** | No (Black-box) |
| **Proposed: Qwen2.5-7B (SFT+GRPO)** | 32.00 *(54.52 on valid outputs)* | 27.57 | 9.66 | 45.48 | 40.73 | **Yes (CoT Rationale)** |

### 3.2. Chain-of-Thought Rationale Quality (Across $N_{\text{valid}} = 326$ Valid Outputs)
* **Average Rationale Length:** `90.0 words` (`4.54 sentences` per response).
* **CoT Transition Rate:** **`76.07%`** (structured step-by-step discourse markers).
* **Forensic Evidence Rate:** **`95.71%`** (explicit citation of text-specific linguistic cues, such as repetitive phrasing in AI text and colloquial contractions or conversational discourse markers in human writing).

---

## 4. Author

* **Author:** Huynh Thanh Tra — B.Sc. in Data Science, Department of Mathematics and Statistics, Quy Nhon University.
* **Date:** November 2026.
