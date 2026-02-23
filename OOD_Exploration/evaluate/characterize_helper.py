import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Global config
with open("./configs/config_training.yaml", "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}

LABEL_ENCODING = config_training["plant_label_encoding"]
LABEL_DECODING = {v: k for k, v in LABEL_ENCODING.items()}


def generate_confusion_matrix(df, visualizations_dir, outputs_or_logits="outputs"):
    def get_cmat_labels(true_label, pred_label, scores, threshold=0.5):
        final_true_label = true_label
        if true_label not in LABEL_ENCODING.keys():
            final_true_label = "OOD"
        final_pred_label = pred_label
        if max(scores) < threshold:
            final_pred_label = "OOD"
        return final_true_label, final_pred_label

    thresholds = [0.5, 0.7, 0.9]
    fig, axes = plt.subplots(3, 1, figsize=(12, 30))

    for ax, thresh in zip(axes, thresholds):
        df[["final_true_label", "final_pred_label"]] = df.apply(
            lambda row: get_cmat_labels(
                row["true_label"], row["predicted_label"], row[outputs_or_logits], threshold=thresh
            ),
            axis=1,
            result_type="expand",
        )

        labels = sorted(LABEL_ENCODING.keys()) + ["OOD"]

        cm = confusion_matrix(df["final_true_label"], df["final_pred_label"], labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.where(row_sums == 0, 0, cm.astype(float) / np.where(row_sums == 0, 1, row_sums))

        # Build annotation: "count\n(pct%)" for each cell
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = cm_normalized[i, j] * 100
                if row_sums[i, 0] == 0:
                    annot[i, j] = "0\n(N/A)"
                else:
                    annot[i, j] = f"{count}\n({pct:.1f}%)"

        sns.heatmap(
            cm_normalized,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"Confusion Matrix — Threshold: {thresh}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(f"Confusion Matrices at Different OOD Thresholds for {outputs_or_logits}", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    save_path = os.path.join(visualizations_dir, f"confusion_matrices_{outputs_or_logits}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved confusion matrices to {save_path}")
