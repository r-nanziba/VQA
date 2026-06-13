"""
plot_thesis_figures_extra.py
============================
Run in Google Colab. Upload results_bert_vit.csv and results_roberta_beit.csv when prompted.

Generates:
  - F1 Score by Tumor Class and Question Type
  - Accuracy by Tumor Class (BERT+ViT vs RoBERTa+BEiT)
  - Top-10 Most Frequently Predicted Answers
  - Top-5 Most Common Wrong Predictions per Tumor Class
  - Radar Chart — Overall Model Performance Comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from math import pi
from google.colab import files

# ── 1. Load CSVs ──────────────────────────────────────────────────────────────
print("Upload results_bert_vit.csv")
files.upload()
df_bv = pd.read_csv("results_bert_vit.csv")

print("Upload results_roberta_beit.csv")
files.upload()
df_rb = pd.read_csv("results_roberta_beit.csv")

# ── 2. Derive columns ─────────────────────────────────────────────────────────
binary_qs = {
    'Is there a tumor visible in this MRI scan?',
    'Is this a glioma tumor?',
    'Is this a meningioma tumor?',
    'Is this a pituitary tumor?',
}

for df in [df_bv, df_rb]:
    df["question_type"] = df["question"].apply(
        lambda q: "binary" if q in binary_qs else "open_ended"
    )
    df["tumor_class"] = (
        df["image_id"].str.split("_").str[0]
        .replace("notumor", "no tumor")
    )
    df["correct"] = df["answer"] == df["predicted"]

tumor_classes = ["glioma", "meningioma", "pituitary", "no tumor"]
tc_labels     = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
C_BV = "#E05C5C"
C_RB = "#5B8DB8"

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_f1(df, question_type=None, tumor_class=None):
    sub = df.copy()
    if question_type:
        sub = sub[sub["question_type"] == question_type]
    if tumor_class:
        sub = sub[sub["tumor_class"] == tumor_class]
    if len(sub) == 0:
        return 0.0
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    all_labels = list(set(sub["answer"].tolist() + sub["predicted"].tolist()))
    le.fit(all_labels)
    y_true = le.transform(sub["answer"])
    y_pred = le.transform(sub["predicted"])
    seen   = list(set(y_true))
    return round(f1_score(y_true, y_pred, average="macro",
                          zero_division=0, labels=seen), 4)

def get_acc(df, question_type=None, tumor_class=None):
    sub = df.copy()
    if question_type:
        sub = sub[sub["question_type"] == question_type]
    if tumor_class:
        sub = sub[sub["tumor_class"] == tumor_class]
    if len(sub) == 0:
        return 0.0
    return round((sub["answer"] == sub["predicted"]).mean(), 4)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — F1 Score by Tumor Class and Question Type
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("F1 Score by Tumor Class and Question Type",
             fontsize=14, fontweight="bold")

x     = np.arange(len(tumor_classes))
width = 0.35

for ax, df, title in zip(axes, [df_bv, df_rb], ["BERT + ViT", "RoBERTa + BEiT"]):
    f1_bin = [get_f1(df, question_type="binary",     tumor_class=c) for c in tumor_classes]
    f1_oe  = [get_f1(df, question_type="open_ended", tumor_class=c) for c in tumor_classes]

    b1 = ax.bar(x - width/2, f1_bin, width, label="Binary",     color="#4CAF50", alpha=0.85)
    b2 = ax.bar(x + width/2, f1_oe,  width, label="Open-Ended", color="#FF9800", alpha=0.85)
    ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Tumor Class")
    ax.set_ylabel("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(tc_labels)
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("f1_per_class.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: f1_per_class.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Accuracy by Tumor Class (both models, side by side)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Overall Accuracy by Tumor Class",
             fontsize=14, fontweight="bold")

bv_acc = [get_acc(df_bv, tumor_class=c) * 100 for c in tumor_classes]
rb_acc = [get_acc(df_rb, tumor_class=c) * 100 for c in tumor_classes]

b1 = ax.bar(x - width/2, bv_acc, width, label="BERT + ViT",     color=C_BV, alpha=0.85)
b2 = ax.bar(x + width/2, rb_acc, width, label="RoBERTa + BEiT", color=C_RB, alpha=0.85)
ax.bar_label(b1, fmt="%.1f%%", padding=3, fontsize=9)
ax.bar_label(b2, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_xlabel("Tumor Class")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels(tc_labels)
ax.set_ylim(0, 120)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("acc_by_tumor_class.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: acc_by_tumor_class.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Top-10 Predicted Answers (Prediction Distribution)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Top-10 Most Frequently Predicted Answers",
             fontsize=14, fontweight="bold")

for ax, df, title, color in zip(axes,
                                 [df_bv, df_rb],
                                 ["BERT + ViT", "RoBERTa + BEiT"],
                                 [C_BV, C_RB]):
    top10 = df["predicted"].value_counts().head(10)
    bars  = ax.barh(top10.index[::-1], top10.values[::-1], color=color, alpha=0.85)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Prediction Count")
    ax.set_ylabel("Predicted Answer")
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("prediction_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: prediction_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Error Analysis: Top-5 Wrong Predictions per Tumor Class
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Top-5 Most Common Wrong Predictions per Tumor Class",
             fontsize=14, fontweight="bold")

for row, (df, model_name) in enumerate([(df_bv, "BERT + ViT"),
                                         (df_rb, "RoBERTa + BEiT")]):
    wrong = df[df["correct"] == False]
    for col, tc in enumerate(tumor_classes):
        ax     = axes[row][col]
        subset = wrong[wrong["tumor_class"] == tc]

        if len(subset) == 0:
            ax.text(0.5, 0.5, "No errors", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(f"{model_name}\n{tc_labels[col]}", fontsize=9)
            ax.axis("off")
            continue

        top5  = subset["predicted"].value_counts().head(5)
        color = C_BV if row == 0 else C_RB
        bars  = ax.barh(top5.index[::-1], top5.values[::-1], color=color, alpha=0.85)
        ax.bar_label(bars, padding=2, fontsize=8)
        ax.set_title(f"{model_name}\n{tc_labels[col]}", fontsize=9)
        ax.set_xlabel("Count", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("error_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: error_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Radar Chart: Overall Model Performance
# ══════════════════════════════════════════════════════════════════════════════
categories = ["Overall\nAccuracy", "Binary\nAccuracy", "Open-Ended\nAccuracy",
              "Glioma\nAccuracy", "Meningioma\nAccuracy", "Pituitary\nAccuracy",
              "No Tumor\nAccuracy"]

def get_radar_values(df):
    return [
        get_acc(df) * 100,
        get_acc(df, question_type="binary") * 100,
        get_acc(df, question_type="open_ended") * 100,
        get_acc(df, tumor_class="glioma") * 100,
        get_acc(df, tumor_class="meningioma") * 100,
        get_acc(df, tumor_class="pituitary") * 100,
        get_acc(df, tumor_class="no tumor") * 100,
    ]

vals_bv = get_radar_values(df_bv)
vals_rb = get_radar_values(df_rb)

N      = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

vals_bv_plot = vals_bv + vals_bv[:1]
vals_rb_plot = vals_rb + vals_rb[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.suptitle("Radar Chart — Overall Model Performance Comparison",
             fontsize=13, fontweight="bold", y=1.02)

ax.plot(angles, vals_bv_plot, "o-", linewidth=2, color=C_BV, label="BERT + ViT")
ax.fill(angles, vals_bv_plot, alpha=0.15, color=C_BV)
ax.plot(angles, vals_rb_plot, "o-", linewidth=2, color=C_RB, label="RoBERTa + BEiT")
ax.fill(angles, vals_rb_plot, alpha=0.15, color=C_RB)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

plt.tight_layout()
plt.savefig("radar_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: radar_chart.png")


# ══════════════════════════════════════════════════════════════════════════════
# Download all
# ══════════════════════════════════════════════════════════════════════════════
for fname in [
    "f1_per_class.png",
    "acc_by_tumor_class.png",
    "prediction_distribution.png",
    "error_analysis.png",
    "radar_chart.png",
]:
    files.download(fname)

print("\nAll figures downloaded.")
