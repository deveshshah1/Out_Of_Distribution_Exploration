import yaml
import os
import pandas as pd
from characterize_helper import (
    generate_confusion_matrix,
    plot_histogram_confidence_by_class,
    plot_umap,
)

# Global config
with open("./configs/config_training.yaml", "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}


def characterize(ckpt_to_use, model_dataset_name):
    # Find correct ckpt for given model run
    model_dir = config_training["experiment_details"]["model_dir"]
    exp_dir = config_training["experiment_details"]["experiment_name"]
    results_dir = os.path.join(model_dir, exp_dir, f"predictions{ckpt_to_use}")

    # Load the dataset
    df = pd.read_parquet(
        os.path.join(results_dir, f"{model_dataset_name}_predictions.parquet")
    )
    df = df[df["stage"] == "test"]

    # Create the visualizations directory
    visualizations_dir = os.path.join(
        model_dir, exp_dir, f"benchmarks{ckpt_to_use}", model_dataset_name
    )
    os.makedirs(visualizations_dir, exist_ok=True)

    # Create visualizations
    generate_confusion_matrix(df, visualizations_dir, outputs_or_logits="outputs")
    if model_dataset_name == "plantpathology":
        plot_histogram_confidence_by_class(
            df, visualizations_dir, outputs_or_logits="outputs"
        )
        plot_umap(df, visualizations_dir, outputs_or_logits="outputs")
    else:
        df_plantpathology = pd.read_parquet(
            os.path.join(results_dir, "plantpathology_predictions.parquet")
        )
        plot_histogram_confidence_by_class(
            df_plantpathology,
            visualizations_dir,
            outputs_or_logits="outputs",
            ood_df=df,
            ood_name=model_dataset_name,
        )
        plot_umap(
            df_plantpathology,
            visualizations_dir,
            outputs_or_logits="outputs",
            ood_df=df,
            ood_name=model_dataset_name,
        )


if __name__ == "__main__":
    datasets_of_interest = [
        "plantpathology",
        "stanfordcars",
        "flowers102",
        "dtd",
    ]
    all_ckpts = ["_best_val_loss", "_best_train_loss", "_best_val_balanced_accuracy"]

    for ckpt in all_ckpts:
        for model_dataset_name in datasets_of_interest:
            characterize(
                ckpt_to_use=ckpt,
                model_dataset_name=model_dataset_name,
            )

    print("Done characterizing the predictions!")
