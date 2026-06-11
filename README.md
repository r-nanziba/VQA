# Dual-Stream Transformer-Based VQA Framework for Brain Tumor MRI Analysis

A Visual Question Answering (VQA) framework that combines transformer-based text and image encoders to interpret brain tumor MRI scans through natural language questions.

## Project Overview

This project was developed as part of a Final Year Project (FYP) at Universiti Teknologi Malaysia (UTM), Malaysia-Japan International Institute of Technology (MJIIT).

The framework enables healthcare professionals to query brain MRI scans using simple natural language questions and receive accurate, contextually relevant answers — addressing the global shortage of radiological expertise.

## Models Implemented

| Model Combination | Text Encoder | Image Encoder |
|---|---|---|
| BERT + ViT | `bert-base-uncased` | `google/vit-base-patch16-224-in21k` |
| RoBERTa + BEiT | `roberta-base` | `microsoft/beit-base-patch16-224-pt22k-ft22k` |

## Dataset

- **Source:** Kaggle Brain Tumor MRI Dataset
- **Classes:** Glioma, Meningioma, Pituitary Tumor, No Tumor
- **Images:** 1,000 MRI images (250 per class)
- **QA Pairs:** 4,000 question-answer pairs (16 questions per image; mix of Yes/No and What/Where)
- **Split:** 3,200 training / 800 evaluation (80/20)

## Results

| Metric | BERT + ViT | RoBERTa + BEiT |
|---|---|---|
| Best Accuracy | 82.75% | 82.38% |
| Best WUPS | 0.8280 | 0.8247 |
| Best F1 | 0.1007 | 0.1047 |
| Best Epoch | 3.5 | 3.5 |
| Training Time | ~51 min | ~86 min |

## Requirements

- Python 3.8+
- PyTorch (CUDA 12.4+)
- Transformers (HuggingFace)
- See `requirements.txt` for the full list

## Setup

```bash
git clone https://github.com/r-nanziba/VQA.git
cd VQA

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Folder Structure

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

## Training

```bash
python src/main.py --config params.yaml
```

## Inference

```bash
python src/inference.py --config params.yaml \
  --img_path dataset/images/glioma_001.jpg \
  --question "Is there a tumor visible in this MRI scan?"
```

## Supervisor

Prof. Dr. Hamido Fujita — Universiti Teknologi Malaysia
