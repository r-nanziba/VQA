"""
plot_thesis_figures.py
======================
Run in Google Colab. Upload results_bert_vit.csv and results_roberta_beit.csv first.

Generates:
  - Figure 5.2  : Dual-Stream Performance Summary (per-category + by question type)
  - Figure 5.3  : Open-Ended Accuracy by Tumor Category
  - Figure 5.4  : Confusion Matrix — Binary Questions (both models)
  - Figure 5.5  : Confusion Matrix — Open-Ended Questions (both models)

All figures are saved as high-res PNG files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── 1. Load CSVs ──────────────────────────────────────────────────────────────
from google.colab import files

print("Upload results_bert_vit.csv")
uploaded = files.upload()
df_bv = pd.read_csv("results_bert_vit.csv")

print("Upload results_roberta_beit.csv")
uploaded = files.upload()
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

tumor_classes   = ["glioma", "meningioma", "pituitary", "no tumor"]
question_types  = ["binary", "open_ended"]

# ── 3. Colour palette ─────────────────────────────────────────────────────────
C_BV = "#E05C5C"   # red  — BERT+ViT
C_RB = "#5B8DB8"   # blue — RoBERTa+BEiT

# ── Helper ────────────────────────────────────────────────────────────────────
def acc(df, **filters):
    subset = df.copy()
    for col, val in filters.items():
        subset = subset[subset[col] == val]
    if len(subset) == 0:
        return 0.0
    return round((subset["answer"] == subset["predicted"]).mean() * 100, 1)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5.2 — Dual-Stream Performance Summary
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Figure 5.2: Dual-Stream VQA Framework — Performance Summary",
             fontsize=14, fontweight="bold", y=1.01)

x      = np.arange(len(tumor_classes))
width  = 0.35
labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# Left: per-category overall accuracy
ax = axes[0]
bv_cat = [acc(df_bv, tumor_class=c) for c in tumor_classes]
rb_cat = [acc(df_rb, tumor_class=c) for c in tumor_classes]

bars1 = ax.bar(x - width/2, bv_cat, width, label="BERT+ViT",      color=C_BV, alpha=0.85)
bars2 = ax.bar(x + width/2, rb_cat, width, label="RoBERTa+BEiT",  color=C_RB, alpha=0.85)
ax.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=9)
ax.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_title("Per-Category Overall Accuracy", fontsize=12)
ax.set_xlabel("Tumor Category"); ax.set_ylabel("Overall Accuracy (%)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 120); ax.legend(); ax.grid(axis="y", alpha=0.3)

# Right: accuracy by question type
ax = axes[1]
qt_labels = ["Binary\n(Yes/No)", "Open-Ended\n(What/Where)"]
bv_qt = [acc(df_bv, question_type=q) for q in question_types]
rb_qt = [acc(df_rb, question_type=q) for q in question_types]
x2 = np.arange(len(question_types))

bars3 = ax.bar(x2 - width/2, bv_qt, width, label="BERT+ViT",     color=C_BV, alpha=0.85)
bars4 = ax.bar(x2 + width/2, rb_qt, width, label="RoBERTa+BEiT", color=C_RB, alpha=0.85)
ax.bar_label(bars3, fmt="%.1f%%", padding=3, fontsize=9)
ax.bar_label(bars4, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_title("Accuracy by Question Type", fontsize=12)
ax.set_xlabel("Question Type"); ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x2); ax.set_xticklabels(qt_labels)
ax.set_ylim(0, 120); ax.legend(); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("fig5_2_performance_updated.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig5_2_performance_updated.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5.3 — Open-Ended Accuracy by Tumor Category
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Figure 5.3: Open-Ended (What/Where) Question Accuracy by Tumor Category",
             fontsize=13, fontweight="bold")

bv_oe = [acc(df_bv, question_type="open_ended", tumor_class=c) for c in tumor_classes]
rb_oe = [acc(df_rb, question_type="open_ended", tumor_class=c) for c in tumor_classes]

bars1 = ax.bar(x - width/2, bv_oe, width, label="BERT+ViT",     color=C_BV, alpha=0.85)
bars2 = ax.bar(x + width/2, rb_oe, width, label="RoBERTa+BEiT", color=C_RB, alpha=0.85)
ax.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=9)
ax.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)
ax.set_xlabel("Tumor Category"); ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 120); ax.legend(); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("fig5_3_openended_updated.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig5_3_openended_updated.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5.4 — Confusion Matrix: Binary Questions
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Figure 5.4: Confusion Matrix — Binary Questions",
             fontsize=14, fontweight="bold")

for ax, df, title in zip(axes,
                          [df_bv, df_rb],
                          ["BERT + ViT", "RoBERTa + BEiT"]):
    sub = df[df["question_type"] == "binary"]
    labels_seen = sorted(sub["answer"].unique())
    cm = confusion_matrix(sub["answer"], sub["predicted"], labels=labels_seen)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_seen)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation=45)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")

plt.tight_layout()
plt.savefig("fig5_4_cm_binary.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig5_4_cm_binary.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5.5 — Confusion Matrix: Open-Ended Questions
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Figure 5.5: Confusion Matrix — Open-Ended Questions",
             fontsize=14, fontweight="bold")

for ax, df, title in zip(axes,
                          [df_bv, df_rb],
                          ["BERT + ViT", "RoBERTa + BEiT"]):
    sub = df[df["question_type"] == "open_ended"]
    # Use only labels that appear as true answers (keeps matrix readable)
    labels_seen = sorted(sub["answer"].unique())
    cm = confusion_matrix(sub["answer"], sub["predicted"], labels=labels_seen)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_seen)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=90)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.tick_params(axis="both", labelsize=7)

plt.tight_layout()
plt.savefig("fig5_5_cm_openended.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: fig5_5_cm_openended.png")


# ══════════════════════════════════════════════════════════════════════════════
# Download all figures
# ══════════════════════════════════════════════════════════════════════════════
for fname in [
    "fig5_2_performance_updated.png",
    "fig5_3_openended_updated.png",
    "fig5_4_cm_binary.png",
    "fig5_5_cm_openended.png",
]:
    files.download(fname)

print("\nAll figures downloaded.")
