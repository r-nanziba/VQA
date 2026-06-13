# 🧠 Dual-Stream Transformer-Based VQA Framework for Brain Tumor MRI Analysis

A Visual Question Answering (VQA) framework that combines transformer-based text and image encoders to interpret brain tumor MRI scans through natural language questions.

## 📌 Project Overview

This project was developed as part of a Final Year Project (FYP2) at **Universiti Teknologi Malaysia (UTM)**, Malaysia-Japan International Institute of Technology (MJIIT), under the supervision of **Prof. Dr. Hamido Fujita**.

The framework enables healthcare professionals to query brain MRI scans using simple natural language questions and receive accurate, contextually relevant answers — addressing the global shortage of radiological expertise.

---

## 🏗️ System Architecture

The framework consists of 5 components:

1. **Visual Features Extractor** — ViT / BEiT image encoder
2. **Textual Features Extractor** — BERT / RoBERTa text encoder
3. **Cross-Modal Feature Attention Module (CFAM)** — MCAoA layers for cross-modal alignment
4. **Feature Fusion Module** — element-wise multiplication of visual and textual features
5. **Answer Generation Module** — classification over a fixed answer space

---

## 🤖 Models Implemented

| Model Combination | Text Encoder | Image Encoder |
|---|---|---|
| BERT + ViT | `bert-base-uncased` | `google/vit-base-patch16-224-in21k` |
| RoBERTa + BEiT | `roberta-base` | `microsoft/beit-base-patch16-224-pt22k-ft22k` |

---

## 📂 Dataset

| Property | Details |
|---|---|
| Source | Kaggle Brain Tumor MRI Dataset |
| Classes | Glioma, Meningioma, Pituitary Tumor, No Tumor |
| Images used | 1,000 MRI images (250 per class) |
| QA pairs | 4,000 total — 16 questions per image |
| Question types | 8 binary (Yes/No) + 8 open-ended (What/Where) per image |
| Split | 3,200 training / 800 evaluation (80/20) |
| Image resolution | 224×224 pixels |
| Answer space | 175 unique answers |

QA pairs were generated using Claude AI and verified manually.

---

## ⚙️ Training Details

| Parameter | Value |
|---|---|
| Epochs | 5 (evaluated every 0.5 epoch) |
| Learning rate | 4.5e-5 (linear decay) |
| Batch size | 16 |
| Best checkpoint | Epoch 3.5 (both models) |
| Training time | BERT+ViT ~51 min \| RoBERTa+BEiT ~86 min |
| Training speed | BERT+ViT 5.22 samples/sec \| RoBERTa+BEiT 3.10 samples/sec |
| Inference speed | BERT+ViT 22.32 it/sec \| RoBERTa+BEiT 21.22 it/sec |
| Platform | Windows, local GPU (CUDA 12.4) |

---

## 📊 Results

### Overall Performance (N=800, Epoch 3.5 — Best Checkpoint)

| Metric | BERT + ViT | RoBERTa + BEiT |
|---|---|---|
| **Accuracy** | **82.75%** | **82.50%** |
| WUPS | 0.8281 | 0.8260 |
| F1 (macro) | 0.1007 | 0.1054 |
| Training Time | ~51 min | ~86 min |

> ℹ️ **Note on F1:** Macro F1 is computed over all answer classes present in the evaluation set. The low overall F1 (~0.10) is a known artifact of large open-ended VQA answer spaces — binary questions achieve perfect F1=1.0, while open-ended questions have many minority answer classes that are rarely predicted, pulling the macro average down. This is discussed as a limitation in the thesis.

---

### By Question Type

| Question Type | N | BERT+ViT Acc | RoBERTa+BEiT Acc |
|---|---|---|---|
| Binary (Yes/No) | 391 | **100.00%** | **100.00%** |
| Open-Ended (What/Where) | 409 | 66.26% | 65.77% |

---

### By Tumor Class

| Class | N | BERT+ViT Acc | BERT+ViT F1 | RoBERTa+BEiT Acc | RoBERTa+BEiT F1 |
|---|---|---|---|---|---|
| Glioma | 185 | 73.51% | 0.0578 | 75.68% | 0.0648 |
| Meningioma | 201 | 71.14% | 0.0635 | 69.15% | 0.0679 |
| Pituitary | 191 | 83.77% | 0.2257 | 82.72% | 0.2273 |
| No Tumor | 223 | **100.00%** | 1.0000 | **100.00%** | 1.0000 |

---

### Open-Ended Questions — By Tumor Class

| Class | N | BERT+ViT Acc | RoBERTa+BEiT Acc |
|---|---|---|---|
| Glioma | 92 | 46.74% | 51.09% |
| Meningioma | 111 | 47.75% | 44.14% |
| Pituitary | 90 | 65.56% | 63.33% |
| No Tumor | 116 | **100.00%** | **100.00%** |

---

## 🔍 Key Findings

- ✅ **Binary questions:** Perfect 100% accuracy across all tumor classes for both models
- ⚠️ **Open-ended questions** are the main challenge — meningioma is the hardest class (~44–48%), no tumor is easiest (100%)
- ⚡ **BERT+ViT trains ~40% faster** than RoBERTa+BEiT with comparable final accuracy — preferred for efficiency
- 🎯 **RoBERTa+BEiT** produces marginally more anatomically precise location answers (e.g. *"sellar and suprasellar region"* vs *"sellar region"*)
- 📉 Both models converge at **epoch 3.5**; performance plateaus slightly after that
- 🏥 Both models reliably identify the **absence of tumors** (No Tumor class: 100% accuracy)

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch (CUDA 12.4+)
- HuggingFace Transformers
- See `requirements.txt` for the full list

---

## 🚀 Setup

```bash
git clone https://github.com/r-nanziba/VQA.git
cd VQA

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## 📁 Folder Structure

```
VQA/
├── dataset/
│   ├── images/              ← 1,000 MRI images (flat, no subfolders)
│   ├── data_train.csv       ← 3,200 training QA pairs
│   ├── data_eval.csv        ← 800 evaluation QA pairs
│   └── answer_space.txt     ← 175 unique answers
├── src/
│   ├── main.py              ← training script
│   └── inference.py         ← inference script
├── params.yaml              ← all configuration
└── requirements.txt
```

---

## 🏋️ Training

```bash
# BERT + ViT
python src/main.py --config params.yaml

# RoBERTa + BEiT
# Update model.text_encoder and model.image_encoder in params.yaml first
python src/main.py --config params.yaml
```

---

## 🔮 Inference

```bash
python src/inference.py --config params.yaml \
  --img_path dataset/images/glioma_001.jpg \
  --question "Is there a tumor visible in this MRI scan?"
```

---

## 📖 Reference

The primary benchmark this study extends:

> Shehzad, F., Minutolo, A., Esposito, M., Fujita, H. and Aljuaid, H. (2024). *"Brain Tumor MRI Interpretation: Towards a Benchmark for Medical Visual Question Answering"*

---

## 👩‍💻 Author

**Rifah Nanziba** — Final Year Software Engineering Student, UTM MJIIT

**Supervisor:** Prof. Dr. Hamido Fujita — Universiti Teknologi Malaysia
