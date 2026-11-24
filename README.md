# Explainable AI-Generated Text Detection via Rationale Distillation and GRPO
### Phát hiện văn bản do AI tạo ra có khả năng giải thích thông qua Chưng cất lập luận và Học tăng cường GRPO

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-PEFT%20%7C%20TRL-yellow.svg)](https://huggingface.co/)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle%20GPU%20T4%20x2-20BEFF.svg)](https://www.kaggle.com/)

## 1. Giới thiệu tổng quan (Overview)

Các mô hình phát hiện văn bản do AI tạo ra truyền thống (như `RoBERTa-large`) thường đạt độ chính xác phân loại cao nhưng hoạt động như một **"hộp đen" (black-box)**, chỉ đưa ra xác suất vô hướng mà không giải thích căn cứ ngôn ngữ học đằng sau quyết định.

Dự án này xây dựng quy trình phát hiện văn bản AI **có khả năng giải thích (Explainable AI-Generated Text Detection)** trên **11 miền thể loại văn bản** dựa trên khung dữ liệu chuẩn **READ** (hợp nhất từ **MAGE** và **L2R**). Mô hình đề xuất **`Qwen2.5-7B-Instruct` (SFT + GRPO)** được huấn luyện qua hai giai đoạn:
1. **Pha 1 — Supervised Fine-Tuning (SFT):** Chưng cất chuỗi lập luận pháp y văn bản (*Chain-of-Thought Rationale*) từ mô hình giáo viên (`Teacher`) giúp mô hình học cách phân tích phong cách, nhịp điệu câu, tính lặp từ vựng và cấu trúc đầu ra JSON.
2. **Pha 2 — Group Relative Policy Optimization (GRPO):** Tối ưu hóa học tăng cường không cần mạng Critic với hàm thưởng kép định hình cấu trúc và độ chính xác phán quyết.

---

## 2. Cấu trúc Repository (Repository Structure)

```text
explainable-ai-text-detection/
├── README.md                                         # Tài liệu tổng quan dự án & tóm tắt kết quả
├── notebooks/                                        # Toàn bộ mã nguồn thực nghiệm (kèm log đầu ra)
│   ├── 01a_data_preparation_splits.ipynb             # Tiền xử lý, hợp nhất dữ liệu & chia tập cân bằng 11 miền
│   ├── 02_teacher_rationale_distillation.ipynb       # Chưng cất chuỗi lập luận từ Teacher Qwen2.5-7B-Instruct
│   ├── 03a_baseline_roberta_training.ipynb           # Huấn luyện & đánh giá mô hình đối chứng RoBERTa-large
│   ├── 03b_student_sft_training.ipynb                # Pha 1: Tinh chỉnh có giám sát (QLoRA SFT) cho Student
│   ├── 03c_student_grpo_training.ipynb               # Pha 2: Tối ưu hóa học tăng cường GRPO cho Student
│   ├── 03d_qwen_fast_eval.ipynb                      # Suy luận & trích xuất lập luận trên tập kiểm thử 11 miền
│   └── 04_evaluation_and_explainability_analysis.ipynb # Tổng hợp bảng số liệu, độ đo CoT & xuất biểu đồ 300 DPI
├── results/                                          # Bảng số liệu & biểu đồ thực nghiệm xuất từ Notebook 04
│   ├── table1_overall_metrics.csv                    # Bảng 1: So sánh hiệu năng tổng quát
│   ├── table2_domain_breakdown.csv                   # Bảng 2: Phân rã hiệu năng theo 11 miền
│   ├── table3_explainability_quality.csv             # Bảng 3: Định lượng chất lượng chuỗi lập luận CoT
│   ├── qualitative_rationale_examples.csv            # Các ví dụ lập luận pháp y văn bản tiêu biểu
│   ├── fig1_domain_macro_f1_comparison.png           # Biểu đồ so sánh Macro-F1 trên 11 miền (300 DPI)
│   └── fig2_confusion_matrices.png                   # Ma trận nhầm lẫn của Baseline và Student (300 DPI)
├── report/                                           # Báo cáo hoàn chỉnh (LaTeX & PDF)
│   ├── main_vi.pdf                                   # Bản báo cáo tiếng Việt
│   ├── main_en.pdf                                   # Bản báo cáo tiếng Anh
│   └── src/                                          # Mã nguồn LaTeX (.tex) và hình ảnh
└── slides/                                           # Slide thuyết trình tổng kết
```

*(Ghi chú: Các tệp dữ liệu gốc, dữ liệu đã chia tập và trọng số mô hình có dung lượng lớn nên không lưu trực tiếp trên GitHub và sẽ được bàn giao riêng theo yêu cầu).*

---

## 3. Tóm tắt Kết quả Thực nghiệm (Key Experimental Results)

Đánh giá đối chứng trên tập kiểm thử phân tầng cân bằng **11 miền ($N = 550$, $50$ mẫu/miền)**:

### 3.1. Hiệu năng Phân loại Tổng quát
| Mô hình | Accuracy (%) | Macro-F1 (%) | AI F1 (%) | Human F1 (%) | Format Error Rate (%) | Khả năng giải thích |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: RoBERTa-large** | **90.91** | **90.89** | **91.32** | **90.46** | **0.00** | Không (Black-box) |
| **Proposed: Qwen2.5-7B (SFT+GRPO)** | 32.00 *(54.52 trên tập hợp lệ)* | 27.57 | 9.66 | 45.48 | 40.73 | **Có (CoT Rationale)** |

### 3.2. Chất lượng Chuỗi Lập luận Giải thích (Trên $N_{\text{valid}} = 326$ phản hồi hợp lệ)
* **Độ dài lập luận trung bình:** `90.0 từ` (`4.54 câu` / phản hồi).
* **Tỷ lệ sử dụng từ nối lập luận từng bước (*CoT Transition Rate*):** **`76.07%`**.
* **Tỷ lệ trích dẫn bằng chứng ngôn ngữ học đặc thù (*Forensic Evidence Rate*):** **`95.71%`** (nhận diện chính xác sự lặp cụm từ máy móc ở văn bản AI và các đặc trưng khẩu ngữ, giọng điệu tự sự tự nhiên ở văn bản do con người viết).

---

## 4. Tác giả (Author)

* **Sinh viên thực hiện:** Huỳnh Thanh Trà — Ngành Khoa học Dữ liệu, Khoa Toán và Thống kê, Trường Đại học Quy Nhơn.
* **Thời gian thực hiện:** Tháng 11/2026.
