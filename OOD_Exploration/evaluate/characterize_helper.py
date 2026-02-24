import yaml
import os
import umap
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
                row["true_label"],
                row["predicted_label"],
                row[outputs_or_logits],
                threshold=thresh,
            ),
            axis=1,
            result_type="expand",
        )

        labels = sorted(LABEL_ENCODING.keys()) + ["OOD"]

        cm = confusion_matrix(
            df["final_true_label"], df["final_pred_label"], labels=labels
        )
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.where(
            row_sums == 0, 0, cm.astype(float) / np.where(row_sums == 0, 1, row_sums)
        )

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
        ax.set_title(
            f"Confusion Matrix — Threshold: {thresh}", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        f"Confusion Matrices at Different OOD Thresholds for {outputs_or_logits}",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    save_path = os.path.join(
        visualizations_dir, f"confusion_matrices_{outputs_or_logits}.png"
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved confusion matrices to {save_path}")


def plot_histogram_confidence_by_class(
    df, visualizations_dir, outputs_or_logits="outputs", ood_df=None, ood_name="OOD"
):
    """
    For each class, plot histogram of max confidence scores split by:
    - Correct: true=a, pred=a
    - Incorrect: true=any, pred=a (but not correct)
    If ood_df is provided, also plots:
    - True in-distribution samples (true=a) vs OOD samples predicted as that class
    """
    classes = sorted(LABEL_ENCODING.keys())
    n_classes = len(classes)

    # 1 col if no ood_df, 2 cols if ood_df provided
    n_cols = 2 if ood_df is not None else 1
    fig, axes = plt.subplots(n_classes, n_cols, figsize=(10 * n_cols, 4 * n_classes))

    # Normalize axes to always be 2D for consistent indexing
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    scores = df[outputs_or_logits].apply(max)
    if ood_df is not None:
        ood_scores = ood_df[outputs_or_logits].apply(max)

    for row_idx, cls in enumerate(classes):
        # --- Left plot: Correct vs Incorrect (existing behavior) ---
        ax_left = axes[row_idx, 0]

        correct_mask = (df["true_label"] == cls) & (df["predicted_label"] == cls)
        incorrect_mask = (df["predicted_label"] == cls) & ~correct_mask

        ax_left.hist(
            scores[correct_mask],
            bins=30,
            alpha=0.6,
            color="steelblue",
            label=f"Correct (true={cls}, pred={cls})",
        )
        ax_left.hist(
            scores[incorrect_mask],
            bins=30,
            alpha=0.6,
            color="tomato",
            label=f"Incorrect (true=any, pred={cls})",
        )

        ax_left.set_title(
            f"Class: {cls}  |  Correct: {correct_mask.sum()}  |  Incorrect: {incorrect_mask.sum()}",
            fontsize=12,
            fontweight="bold",
        )
        ax_left.set_xlabel("Max Confidence Score", fontsize=10)
        ax_left.set_ylabel("Count", fontsize=10)
        ax_left.set_xlim(0, 1)
        ax_left.legend(fontsize=9)
        _add_threshold_lines(ax_left)

        # --- Right plot: In-distribution true samples vs OOD predicted as cls ---
        if ood_df is not None:
            ax_right = axes[row_idx, 1]

            true_cls_mask = df["true_label"] == cls
            ood_pred_cls_mask = ood_df["predicted_label"] == cls

            ax_right.hist(
                scores[true_cls_mask],
                bins=30,
                alpha=0.6,
                color="steelblue",
                label=f"In-dist (true={cls})",
            )
            ax_right.hist(
                ood_scores[ood_pred_cls_mask],
                bins=30,
                alpha=0.6,
                color="darkorange",
                label=f"{ood_name} predicted as {cls}",
            )

            ax_right.set_title(
                f"Class: {cls}  |  In-dist: {true_cls_mask.sum()}  |  {ood_name}→{cls}: {ood_pred_cls_mask.sum()}",
                fontsize=12,
                fontweight="bold",
            )
            ax_right.set_xlabel("Max Confidence Score", fontsize=10)
            ax_right.set_ylabel("Count", fontsize=10)
            ax_right.set_xlim(0, 1)
            ax_right.legend(fontsize=9)
            _add_threshold_lines(ax_right)

    ood_suffix = f" + {ood_name}" if ood_df is not None else ""
    fig.suptitle(
        f"Confidence Score Distributions by Class ({outputs_or_logits}){ood_suffix}",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.97)

    ood_file_suffix = (
        f"_{ood_name.lower().replace(' ', '_')}" if ood_df is not None else ""
    )
    save_path = os.path.join(
        visualizations_dir,
        f"confidence_by_class_{outputs_or_logits}{ood_file_suffix}.png",
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved confidence by class plot to {save_path}")


def _add_threshold_lines(ax):
    ax.axvline(
        x=0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="0.5 threshold",
    )
    ax.axvline(
        x=0.7,
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.7,
        label="0.7 threshold",
    )
    ax.axvline(
        x=0.9,
        color="black",
        linestyle=":",
        linewidth=1,
        alpha=0.7,
        label="0.9 threshold",
    )


def plot_umap(
    df, visualizations_dir, outputs_or_logits="outputs", ood_df=None, ood_name="OOD"
):
    """
    Plot UMAP projections of embeddings colored by true and predicted labels.
    Row 1: All samples colored by label
    Row 2: Samples with confidence < 0.9 grayed out
    Row 3: Samples with confidence < 0.7 grayed out
    Row 4: Samples with confidence < 0.5 grayed out
    Optionally overlay OOD embeddings as X markers using the same color scheme.
    """
    embeddings = np.stack(df["embedding"].values)
    scores = df[outputs_or_logits].apply(max)

    # Fit UMAP on in-distribution embeddings only
    print("Fitting UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    proj = reducer.fit_transform(embeddings)

    # Transform OOD embeddings using the same fitted reducer
    if ood_df is not None:
        ood_embeddings = np.stack(ood_df["embedding"].values)
        ood_proj = reducer.transform(ood_embeddings)
        ood_scores = ood_df[outputs_or_logits].apply(max)

    # Consistent color map across all labels
    all_labels = sorted(
        set(
            df["true_label"].unique().tolist() + df["predicted_label"].unique().tolist()
        )
    )
    color_map = {label: color for label, color in zip(all_labels, plt.cm.tab20.colors)}

    thresholds = [None, 0.9, 0.7, 0.5]
    row_titles = [
        "All Samples",
        "Confidence ≥ 0.9 colored (< 0.9 grayed)",
        "Confidence ≥ 0.7 colored (< 0.7 grayed)",
        "Confidence ≥ 0.5 colored (< 0.5 grayed)",
    ]

    fig, axes = plt.subplots(4, 2, figsize=(20, 32))

    for row_idx, (thresh, row_title) in enumerate(zip(thresholds, row_titles)):
        for col_idx, label_col in enumerate(["true_label", "predicted_label"]):
            ax = axes[row_idx, col_idx]

            if thresh is None:
                # Row 1: color everything normally
                for label in all_labels:
                    mask = df[label_col] == label
                    ax.scatter(
                        proj[mask, 0],
                        proj[mask, 1],
                        c=[color_map[label]],
                        label=label,
                        alpha=0.6,
                        s=10,
                    )

                # Overlay OOD as X's colored by predicted label
                if ood_df is not None:
                    for label in all_labels:
                        ood_mask = ood_df["predicted_label"] == label
                        if ood_mask.sum() == 0:
                            continue
                        ax.scatter(
                            ood_proj[ood_mask, 0],
                            ood_proj[ood_mask, 1],
                            c=[color_map[label]],
                            marker="X",
                            s=40,
                            alpha=0.8,
                            edgecolors="black",
                            linewidths=0.3,
                            label=f"{ood_name}→{label}",
                        )

            else:
                # Gray out low confidence, color high confidence
                low_conf_mask = scores < thresh
                high_conf_mask = ~low_conf_mask

                ax.scatter(
                    proj[low_conf_mask, 0],
                    proj[low_conf_mask, 1],
                    c="lightgray",
                    alpha=0.3,
                    s=10,
                    label=f"Confidence < {thresh}",
                )

                for label in all_labels:
                    mask = high_conf_mask & (df[label_col] == label)
                    if mask.sum() == 0:
                        continue
                    ax.scatter(
                        proj[mask, 0],
                        proj[mask, 1],
                        c=[color_map[label]],
                        label=label,
                        alpha=0.6,
                        s=10,
                    )

                # Overlay OOD as X's, gray if below threshold, colored if above
                if ood_df is not None:
                    ood_low_conf = ood_scores < thresh
                    ood_high_conf = ~ood_low_conf

                    # Gray OOD points below threshold
                    ax.scatter(
                        ood_proj[ood_low_conf, 0],
                        ood_proj[ood_low_conf, 1],
                        c="lightgray",
                        marker="X",
                        s=40,
                        alpha=0.4,
                        edgecolors="gray",
                        linewidths=0.3,
                        label=f"{ood_name} (low conf)",
                    )

                    # Colored OOD points above threshold
                    for label in all_labels:
                        ood_mask = ood_high_conf & (ood_df["predicted_label"] == label)
                        if ood_mask.sum() == 0:
                            continue
                        ax.scatter(
                            ood_proj[ood_mask, 0],
                            ood_proj[ood_mask, 1],
                            c=[color_map[label]],
                            marker="X",
                            s=40,
                            alpha=0.8,
                            edgecolors="black",
                            linewidths=0.3,
                            label=f"{ood_name}→{label}",
                        )

            col_title = "True Labels" if col_idx == 0 else "Predicted Labels"
            ax.set_title(f"{col_title} — {row_title}", fontsize=12, fontweight="bold")
            ax.set_xlabel("UMAP 1", fontsize=10)
            ax.set_ylabel("UMAP 2", fontsize=10)
            ax.legend(
                fontsize=8, markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left"
            )

    ood_suffix = f" + {ood_name}" if ood_df is not None else ""
    fig.suptitle(
        f"UMAP Projections ({outputs_or_logits}){ood_suffix}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    ood_file_suffix = (
        f"_{ood_name.lower().replace(' ', '_')}" if ood_df is not None else ""
    )
    save_path = os.path.join(
        visualizations_dir, f"umap_{outputs_or_logits}{ood_file_suffix}.png"
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved UMAP plot to {save_path}")
