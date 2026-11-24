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
│   ├── 03d_qwen_fast_eval.ipynb                      # Inference
