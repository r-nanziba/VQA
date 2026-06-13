"""
evaluate_breakdown.py
=====================
Run from inside VQA-With-Multimodal-Transformers/ (same folder as params.yaml).

BERT + ViT:
  python evaluate_breakdown.py --checkpoint checkpoint_bert_vit/bert-vit/checkpoint-700 --config params.yaml --output_csv results_bert_vit.csv

RoBERTa + BEiT  (params.yaml must have roberta-base / beit encoder):
  python evaluate_breakdown.py --checkpoint checkpoint/roberta-beit/checkpoint- --config params.yaml --output_csv results_roberta_beit.csv
"""

import argparse, os, json, sys
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor

# ── Exact model class from the repo ─────────────────────────────────────────
class MultimodalVQAModel(nn.Module):
    def __init__(self, num_labels, intermediate_dims, dropout,
                 pretrained_text_name, pretrained_image_name):
        super().__init__()
        self.num_labels          = num_labels
        self.pretrained_text_name  = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name

        self.text_encoder  = AutoModel.from_pretrained(pretrained_text_name)
        self.image_encoder = AutoModel.from_pretrained(pretrained_image_name)

        self.fusion = nn.Sequential(
            nn.Linear(
                self.text_encoder.config.hidden_size +
                self.image_encoder.config.hidden_size,
                intermediate_dims
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(intermediate_dims, num_labels)
        self.criterion  = nn.CrossEntropyLoss()

    def forward(self, input_ids, pixel_values,
                attention_mask=None, token_type_ids=None, labels=None):
        enc_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        enc_img = self.image_encoder(pixel_values=pixel_values, return_dict=True)
        fused   = self.fusion(torch.cat(
            [enc_text['pooler_output'], enc_img['pooler_output']], dim=1
        ))
        logits = self.classifier(fused)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = self.criterion(logits, labels)
        return out


# ── WUPS (same logic as repo's evaluate.py) ──────────────────────────────────
def wup_measure(a, b, similarity_threshold=0.925):
    try:
        from nltk.corpus import wordnet
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except ImportError:
        return 1.0 if a.strip().lower() == b.strip().lower() else 0.0

    if a == b:
        return 1.0
    sa = wordnet.synsets(a, pos=wordnet.NOUN)
    sb = wordnet.synsets(b, pos=wordnet.NOUN)
    if not sa or not sb:
        return 0.0
    best = max((x.wup_similarity(y) or 0.0) for x in sa for y in sb)
    return best if best >= similarity_threshold else best * 0.1


# ── Metrics for a DataFrame slice ────────────────────────────────────────────
def compute_metrics(df_slice):
    if len(df_slice) == 0:
        return {"n": 0, "accuracy": None, "wups": None, "f1": None}

    labels = df_slice["label"].values
    preds  = df_slice["pred_idx"].values

    acc  = float((labels == preds).mean())
    wups = float(np.mean([
        wup_measure(df_slice["answer_space"].iloc[0][l],
                    df_slice["answer_space"].iloc[0][p])
        for l, p in zip(labels, preds)
    ])) if "answer_space" in df_slice.columns else 0.0

    from sklearn.metrics import f1_score
    f1 = float(f1_score(labels, preds, average='macro', zero_division=0, labels=list(set(labels))))

    return {
        "n":        len(df_slice),
        "accuracy": round(acc,  4),
        "wups":     round(wups, 4),
        "f1":       round(f1,   4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--config",        default="params.yaml")
    parser.add_argument("--eval_csv",      default="dataset/data_eval.csv")
    parser.add_argument("--images_folder", default="dataset/images")
    parser.add_argument("--answer_space",  default="dataset/answer_space.txt")
    parser.add_argument("--output_csv",    default="eval_results.csv")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    text_encoder  = config["model"]["text_encoder"]
    image_encoder = config["model"]["image_encoder"]
    inter_dims    = config["model"]["intermediate_dims"]
    dropout       = config["model"]["dropout"]

    # ── Answer space ──────────────────────────────────────────────────────────
    with open(args.answer_space) as f:
        answer_space = [l.strip() for l in f if l.strip()]
    num_labels = len(answer_space)
    ans2idx    = {a: i for i, a in enumerate(answer_space)}
    print(f"Answer space: {num_labels} answers")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build model architecture then load saved weights ─────────────────────
    print(f"Building model: {text_encoder} + {image_encoder}")
    model = MultimodalVQAModel(
        num_labels=num_labels,
        intermediate_dims=inter_dims,
        dropout=dropout,
        pretrained_text_name=text_encoder,
        pretrained_image_name=image_encoder,
    )

    safetensors_file = os.path.join(args.checkpoint, "model.safetensors")
    bin_file = os.path.join(args.checkpoint, "pytorch_model.bin")

    if os.path.exists(safetensors_file):
        from safetensors.torch import load_file
        ckpt_file = safetensors_file
        print(f"Loading weights from: {ckpt_file}")
        state = load_file(ckpt_file, device="cpu")
    elif os.path.exists(bin_file):
        ckpt_file = bin_file
        print(f"Loading weights from: {ckpt_file}")
        state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"No checkpoint weights found in {args.checkpoint}")

    # state could be a raw state_dict OR a dict with a 'state_dict' key
    if isinstance(state, dict) and not any(k.startswith("text_encoder") for k in state):
        # It's a plain dict but not a state_dict — check for nested key
        if "state_dict" in state:
            state = state["state_dict"]
        else:
            print("WARNING: checkpoint keys don't look like model weights.")
            print("First 5 keys:", list(state.keys())[:5])
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # ── Processors ────────────────────────────────────────────────────────────
    tokenizer         = AutoTokenizer.from_pretrained(text_encoder)
    feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder)
    tok_cfg           = config.get("tokenizer", {})

    # ── Load eval CSV ─────────────────────────────────────────────────────────
    df = pd.read_csv(args.eval_csv)

    # Derive label index from answer column
    df["label"] = df["answer"].apply(
        lambda a: ans2idx.get(a.replace(" ", "").split(",")[0],
                              ans2idx.get(a, 0))
    )

    # Derive question_type
    binary_qs = {
        'Is there a tumor visible in this MRI scan?',
        'Is this a glioma tumor?',
        'Is this a meningioma tumor?',
        'Is this a pituitary tumor?',
    }
    df["question_type"] = df["question"].apply(
        lambda q: "binary" if q in binary_qs else "open_ended"
    )

    # Derive tumor_class from image_id
    df["tumor_class"] = (
        df["image_id"].str.split("_").str[0]
        .replace("notumor", "no tumor")
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    pred_indices = []
    pred_answers = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        # Try .jpg then .png
        img_path = os.path.join(args.images_folder, f"{row['image_id']}.jpg")
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".png")

        try:
            image = Image.open(img_path).convert("RGB")
            img_feats = feature_extractor(images=image, return_tensors="pt")
            img_feats = {k: v.to(device) for k, v in img_feats.items()}

            txt_feats = tokenizer(
                row["question"],
                padding=tok_cfg.get("padding", "longest"),
                max_length=tok_cfg.get("max_length", 24),
                truncation=tok_cfg.get("truncation", True),
                return_token_type_ids=tok_cfg.get("return_token_type_ids", True),
                return_attention_mask=tok_cfg.get("return_attention_mask", True),
                return_tensors="pt",
            )
            txt_feats = {k: v.to(device) for k, v in txt_feats.items()}

            with torch.no_grad():
                out      = model(**txt_feats, **img_feats)
                pred_idx = out["logits"].argmax(-1).item()

        except Exception as e:
            print(f"\nError on {row['image_id']}: {e}")
            pred_idx = 0

        pred_indices.append(pred_idx)
        pred_answers.append(answer_space[pred_idx])

    df["pred_idx"]  = pred_indices
    df["predicted"] = pred_answers

    # Add answer_space column (needed by compute_metrics — store as object ref)
    df["answer_space"] = [answer_space] * len(df)

    # Save full results
    df.drop(columns=["answer_space"]).to_csv(args.output_csv, index=False)
    print(f"\nPredictions saved to: {args.output_csv}")

    # ── Print & save breakdowns ───────────────────────────────────────────────
    def show(label, subset):
        m = compute_metrics(subset)
        print(f"  {label:<35s}  N={m['n']:4d}  "
              f"Acc={m['accuracy']}  WUPS={m['wups']}  F1={m['f1']}")
        return m

    print("\n" + "="*70)
    print("RESULTS BREAKDOWN")
    print("="*70)

    summary = {}

    print("\n── OVERALL ─────────────────────────────────────────────────────────")
    summary["overall"] = show("Overall", df)

    print("\n── BY QUESTION TYPE ────────────────────────────────────────────────")
    summary["by_question_type"] = {}
    for qt in ["binary", "open_ended"]:
        summary["by_question_type"][qt] = show(qt, df[df["question_type"] == qt])

    print("\n── BY TUMOR CLASS ──────────────────────────────────────────────────")
    summary["by_tumor_class"] = {}
    for cls in ["glioma", "meningioma", "pituitary", "no tumor"]:
        summary["by_tumor_class"][cls] = show(cls, df[df["tumor_class"] == cls])

    print("\n── BINARY QUESTIONS — PER TUMOR CLASS ──────────────────────────────")
    summary["binary_per_class"] = {}
    for cls in ["glioma", "meningioma", "pituitary", "no tumor"]:
        subset = df[(df["question_type"] == "binary") & (df["tumor_class"] == cls)]
        summary["binary_per_class"][cls] = show(f"binary  | {cls}", subset)

    print("\n── OPEN-ENDED QUESTIONS — PER TUMOR CLASS ──────────────────────────")
    summary["openended_per_class"] = {}
    for cls in ["glioma", "meningioma", "pituitary", "no tumor"]:
        subset = df[(df["question_type"] == "open_ended") & (df["tumor_class"] == cls)]
        summary["openended_per_class"][cls] = show(f"open_ended | {cls}", subset)

    # Save summary JSON
    summary_path = args.output_csv.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    print("\nDone! Copy the numbers above into your thesis tables.")


if __name__ == "__main__":
    main()
