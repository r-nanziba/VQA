# 🧠 Dual-Stream Transformer-Based VQA Framework for Brain Tumor MRI Analysis

> *"What type of brain tumor is shown?"* -- Ask the MRI. Get an answer.

A Visual Question Answering (VQA) framework that combines transformer-based text and image encoders to interpret brain tumor MRI scans through natural language questions.

---

## 📌 Project Overview

This project was developed as part of a Final Year Project (FYP) at **Universiti Teknologi Malaysia (UTM)**, Malaysia-Japan International Institute of Technology (MJIIT), under the supervision of **Prof. Dr. Hamido Fujita**.

The framework enables healthcare professionals to query brain MRI scans using simple natural language questions and receive accurate, contextually relevant answers, addressing the global shortage of radiological expertise documented by the World Health Organization.

Two model configurations were implemented and comparatively evaluated under identical training and evaluation conditions: BERT combined with Vision Transformer (ViT-B/16), and RoBERTa combined with BEiT. Rather than reporting accuracy figures alone, this work explicitly examines how the dataset construction process shapes those numbers -- an honest evaluation that is often missing from medical AI benchmarking.

---

## 🏗️ System Architecture

The framework consists of 5 components:

1. 🖼️ **Visual Feature Extractor** -- ViT-B/16 (Configuration 1) or BEiT (Configuration 2) image encoder
2. 📝 **Textual Feature Extractor** -- BERT (Configuration 1) or RoBERTa (Configuration 2) text encoder
3. 🔀 **Cross-Modal Feature Attention Module (CFAM)** -- multi-head cross-attention across 8 attention heads and 4 fusion layers
4. ⚙️ **Feature Fusion Module** -- element-wise multiplication of attended visual and textual representations
5. 🎯 **Answer Classification Module** -- classification over a fixed 175-class answer vocabulary

Both configurations share identical CFAM, fusion, and classification components. Only the visual and textual encoders differ between configurations.

---

## 🤖 Models Implemented

| Configuration | Text Encoder | Image Encoder |
|---|---|---|
| ⚡ Configuration 1: BERT + ViT | `bert-base-uncased` | `google/vit-base-patch16-224-in21k` |
| 🔬 Configuration 2: RoBERTa + BEiT | `roberta-base` | `microsoft/beit-base-patch16-224-pt22k-ft22k` |

Both configurations use standard publicly available pre-trained weights without domain-specific modification, enabling a clean comparison of encoder pre-training strategies under identical conditions.

---

## 📂 Dataset

| Property | Details |
|---|---|
| 🗂️ Source | [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| 🏷️ Classes | Glioma, Meningioma, Pituitary Tumor, No Tumor |
| 🖼️ Images used | 1,000 MRI images (250 per class) |
| ❓ QA pairs | 4,000 total (4 questions per image) |
| 📋 Question types | 2 binary (Yes/No) + 2 open-ended (What/Where) per image |
| ✂️ Split | 3,200 training / 800 evaluation (80/20, random_state=42) |
| 📐 Image resolution | 224x224 pixels (handled by HuggingFace AutoFeatureExtractor at runtime) |
| 🗃️ Answer space | 175 unique answers |

QA pairs were generated with the assistance of Claude (a large language model), operating from per-class templates, and were manually reviewed for medical plausibility. No independent radiologist verification was performed. The binary answers are class-consistent by construction, and open-ended location answers are drawn from a fixed set of canonical phrases per tumor class. Both of these properties have important consequences for interpreting the reported accuracy figures -- see the ⚠️ Dataset Bias section below and the full discussion in the thesis.

---

## ⚙️ Training Details

| Parameter | Value |
|---|---|
| 🔧 Optimizer | AdamW |
| 📉 Learning rate | 4.5e-5 (linear decay) |
| 📦 Batch size | 16 |
| 🔁 Epochs | 5 (evaluated every 0.5 epoch, 10 checkpoints total) |
| ⚖️ Weight decay | 0.0 |
| 💧 Dropout | 0.5 (applied across the fusion module) |
| 📊 Loss function | Cross-entropy |
| 🏆 Best checkpoint | Epoch 3.5 (both configurations, selected on highest validation accuracy and WUPS) |
| ⏱️ Training time | BERT+ViT approx. 51 min (3,065 s) / RoBERTa+BEiT approx. 86 min (5,159 s) |
| 🚀 Training speed | BERT+ViT 5.22 samples/sec / RoBERTa+BEiT 3.10 samples/sec |
| ⚡ Inference speed | BERT+ViT 22.32 it/sec / RoBERTa+BEiT 21.22 it/sec |
| 💻 Hardware | ASUS Vivobook, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), 16 GB RAM |
| 🖥️ OS and runtime | Windows 11, Python 3.11.9, PyTorch 2.6.0+cu124, HuggingFace Transformers 4.38.2 |
| 🔄 Execution | Windows Command Prompt via DVC pipeline (dvc repro) |

---

## 📊 Results

### 🏅 Overall Performance (N=800, Epoch 3.5 Best Checkpoint)

| Metric | ⚡ BERT + ViT | 🔬 RoBERTa + BEiT |
|---|---|---|
| **Overall Accuracy** | **82.75%** | **82.38%** |
| WUPS (threshold 0.9) | 0.8281 | 0.8260 |
| F1 (macro) | 0.1007 | 0.1047 |
| Training Time | approx. 51 min | approx. 86 min |

> ℹ️ **Note on RoBERTa+BEiT accuracy:** A marginal discrepancy exists between the figure logged during training via DVC (82.50%, F1 0.1054) and the figure produced by the post-hoc evaluate_breakdown.py script (82.38%, F1 0.1047). This is expected: the HuggingFace Trainer evaluates in batches during training, while evaluate_breakdown.py processes one sample at a time, producing small floating-point differences. The evaluate_breakdown.py figures (82.38%, F1 0.1047) are used as the primary results throughout the thesis, as they correspond directly to the saved checkpoint used for all reported evaluations.

> ℹ️ **Note on F1:** Macro F1 is computed over all answer classes present in the evaluation set. The low overall F1 (approx. 0.10) is a known characteristic of large open-ended VQA answer spaces. Binary questions achieve a perfect F1 of 1.0 for both configurations, while open-ended questions have many minority answer classes that are rarely predicted, pulling the macro average down significantly. This is not a model failure -- it reflects class imbalance in the open-ended answer distribution and is discussed fully in the thesis.

---

### 📋 Performance by Question Type

| Question Type | N | ⚡ BERT+ViT Acc | ⚡ BERT+ViT WUPS | 🔬 RoBERTa+BEiT Acc | 🔬 RoBERTa+BEiT WUPS |
|---|---|---|---|---|---|
| ✅ Binary (Yes/No) | 391 | **100.00%** | 1.0000 | **100.00%** | 1.0000 |
| 💬 Open-Ended (What/Where) | 409 | 66.26% | 0.6637 | 65.77% | 0.6596 |

---

### 🧬 Performance by Tumor Category (All Question Types)

| Class | N | ⚡ BERT+ViT Acc | ⚡ BERT+ViT F1 | 🔬 RoBERTa+BEiT Acc | 🔬 RoBERTa+BEiT F1 |
|---|---|---|---|---|---|
| 🔴 Glioma | 185 | 73.51% | 0.0578 | 75.68% | 0.0648 |
| 🟠 Meningioma | 201 | 71.14% | 0.0635 | 69.15% | 0.0679 |
| 🟡 Pituitary | 191 | 83.77% | 0.2257 | 82.72% | 0.2273 |
| 🟢 No Tumor | 223 | **100.00%** | 1.0000 | **100.00%** | 1.0000 |

---

### 💬 Open-Ended Questions by Tumor Category

| Class | N | ⚡ BERT+ViT Acc | 🔬 RoBERTa+BEiT Acc |
|---|---|---|---|
| 🔴 Glioma | 92 | 46.74% | 51.09% |
| 🟠 Meningioma | 111 | 47.75% | 44.14% |
| 🟡 Pituitary | 90 | 65.56% | 63.33% |
| 🟢 No Tumor | 116 | **100.00%** | **100.00%** |

---

## ⚠️ Dataset Bias and Honest Interpretation

The headline accuracy figures should be read alongside an understanding of how the dataset was constructed.

**🔲 Binary questions:** The correct answer to every binary question is a deterministic function of the tumor class. For example, "Is this a glioma tumor?" is answered "yes" for every glioma image and "no" for all other classes. A model that learns to classify the tumor class correctly will automatically answer all binary questions correctly, regardless of specific question wording. The 100% binary accuracy reflects this structural property of the dataset rather than unconstrained visual-linguistic reasoning.

**💬 Open-ended questions:** Location answers were drawn from a fixed set of canonical anatomical phrases per tumor class rather than from per-image radiological assessment. The open-ended task is therefore partially reducible to selecting among a handful of class-associated templates. The dominant errors identified in evaluation are confusions between templates within the same tumor class, not random or cross-class failures.

These findings do not invalidate the comparative results between BERT+ViT and RoBERTa+BEiT, since both configurations are affected equally. However, the absolute accuracy figures should not be interpreted as evidence of fine-grained, per-patient radiological reasoning. The primary recommendation for future work is radiologist involvement in QA-pair design and validation.

---

## 🔍 Key Findings

- ✅ **Perfect binary accuracy (100%)** across all tumor classes for both configurations -- this reflects the class-consistent structure of the binary QA pairs rather than unconstrained visual-linguistic reasoning.
- 💬 **Open-ended questions are the real challenge.** Meningioma is the hardest category (44 to 48% open-ended accuracy), consistent with its known morphological variability in MRI. No Tumor achieves 100% because the ground-truth answer is class-invariant by construction.
- ⚡ **BERT+ViT trains ~40% faster** than RoBERTa+BEiT (51 vs 86 minutes) with near-equivalent final accuracy, making it the more practical configuration for this task.
- 🔬 **RoBERTa+BEiT** achieves a marginally higher macro-F1 (0.1047 vs 0.1007) and lower final training loss (0.62 vs 0.69), but these differences do not translate into a meaningful accuracy advantage.
- 📉 **Both models converge at epoch 3.5.** Performance plateaus and shows minor fluctuation beyond that point.
- 🏥 **Zero false positives on No Tumor.** Both models achieve 100% accuracy on the no-tumor category across all question types, confirming the framework never incorrectly flags a healthy scan on this dataset.

---

## 🛠️ Requirements

- 🐍 Python 3.11.9 (developed and tested on this version; 3.8+ may work but is untested)
- 🔥 PyTorch 2.6.0 with CUDA 12.4
- 🤗 HuggingFace Transformers 4.38.2
- 🔄 DVC (for pipeline automation via dvc repro)
- 📦 Additional: NumPy, Pandas, scikit-learn, PyYAML, tqdm, Matplotlib
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
│   ├── images/                  <- 1,000 MRI images (flat folder, no subfolders)
│   ├── data_train.csv           <- 3,200 training QA pairs
│   ├── data_eval.csv            <- 800 evaluation QA pairs
│   └── answer_space.txt         <- 175 unique answers
├── src/
│   ├── process_data.py          <- reads QA pairs, builds answer vocab, performs 80/20 split
│   ├── main.py                  <- training and overall evaluation script (run via DVC)
│   └── inference.py             <- single-sample inference script
├── evaluate_breakdown.py        <- post-hoc per-question-type and per-category evaluation
├── dvc.yaml                     <- DVC pipeline definition
├── params.yaml                  <- all configuration (model, paths, hyperparameters)
└── requirements.txt
```

---

## 🏋️ Training

The training and overall evaluation pipeline is automated using DVC:

```bash
dvc repro
```

This runs process_data.py and main.py sequentially as defined in dvc.yaml, using the settings in params.yaml. To switch between configurations, update `model.text_encoder` and `model.image_encoder` in params.yaml before running.

For manual execution without DVC:

```bash
# Step 1: process data
python src/process_data.py --config params.yaml

# Step 2: train and evaluate
python src/main.py --config params.yaml
```

For the detailed per-question-type and per-category breakdown (used for all results reported in the thesis), run evaluate_breakdown.py manually against the saved checkpoint after training:

```bash
python evaluate_breakdown.py --config params.yaml --checkpoint results/checkpoint-700
```

---

## 🔮 Inference

```bash
python src/inference.py --config params.yaml \
  --img_path dataset/images/glioma_001.jpg \
  --question "Is there a tumor visible in this MRI scan?"
```

Inference runs on the saved best checkpoint (checkpoint-700, epoch 3.5). Expected latency is 1 to 3 seconds per sample on the development machine (NVIDIA GeForce RTX 3050 Laptop GPU).

---

## 📖 Reference

The primary benchmark this study builds on:

> Shehzad, F., Minutolo, A., Esposito, M., Fujita, H. and Aljuaid, H. (2026). "Brain Tumor MRI Interpretation: Towards a Benchmark for Medical Visual Question Answering." In: Advances and Trends in Artificial Intelligence. Theory and Applications. IEA/AIE 2025. Lecture Notes in Computer Science, vol. 15706. Singapore: Springer, pp. 519-530.

Full reference list is available in the thesis document.

---

## 👩‍💻 Author

**Rifah Nanziba** -- Final Year Bachelor of Computer Science (Software Engineering) with Honours, UTM MJIIT

**Supervisor:** Prof. Dr. Hamido Fujita, Universiti Teknologi Malaysia

**Thesis title:** Dual-Stream Transformer-Based VQA Framework for Brain Tumor MRI Analysis: A Comparative Study of BERT+ViT and RoBERTa+BEiT on a Custom Clinical Q&A Dataset
