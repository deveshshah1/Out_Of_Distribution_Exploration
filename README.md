# Out-of-Distribution Detection on Plant Pathology

This repository contains the code for the paper [**"Beyond Toy Benchmarks: 
Evaluating OOD Detection Methods on a Real-World Plant Pathology Dataset"**](https://arxiv.org/abs/2605.08618), 
which presents a systematic evaluation of six OOD detection methods on the 
[Plant Pathology 2021 FGVC8](https://www.kaggle.com/c/plant-pathology-2021-fgvc8) 
dataset.

## Overview

We evaluate six OOD detection methods spanning a progression of assumptions 
about auxiliary data availability at training time:

| Branch | Method | Description |
|--------|---------|-------------|
| `main` | E1: Softmax Baseline | ResNet-18 fine-tuned with cross-entropy loss, MSP scoring |
| `e2_invididual_bce_losses` | E2: Independent BCE | Independent sigmoid heads, no softmax coupling |
| `e3_ood_class` | E3: Explicit OOD Class | 6th class trained on ImageNet-O |
| `e4_outlier_exposure` | E4: Outlier Exposure | Auxiliary uniform distribution loss on OOD data |
| `e5_energy_based_finetuning/inference` | E5: Energy-Based | Post-hoc energy scoring + energy margin fine-tuning |
| `e6_woods` | E6: WOODS | Constrained optimization on unlabeled wild data |

## Datasets

### In-Distribution
- **Plant Pathology 2021** — 15,675 images filtered to 5 single-label disease 
  classes (Scab, Healthy, Frog Eye Leaf Spot, Rust, Powdery Mildew). 
  Download from [Kaggle](https://www.kaggle.com/c/plant-pathology-2021-fgvc8).

### OOD Evaluation (never seen during training)
- **Stanford Cars** — far-OOD, vehicles
- **DTD** — far-OOD, textures
- **Flowers102** — near-OOD, flowers

### Auxiliary OOD Training Data
- **ImageNet-O** — used in E3-E6 for training only. Download from 
  [here](https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar).

## Setup

```bash
git clone https://github.com/deveshshah1/Out_Of_Distribution_Exploration.git
cd Out_Of_Distribution_Exploration
pip install -r requirements.txt
```

To run a specific experiment, checkout the corresponding branch:

```bash
git checkout e4_outlier_exposure
```

## Repository Structure
```
├── dataset/                   # dataset directories (images not included)
│   ├── plantpathology/
│   ├── standfordcars/
│   ├── dtd/
│   ├── flowers102/
│   └── imagenet-o/
├── OOD_Exploration/               
│   ├── evaluate           # characterize scripts to generate plots and statistics
│   ├── pyL_modules.py           # PyTorch Lightning DataModule
│   ├── custom_dataset.py                # Torch dataloader for get_item call during training
│   ├── model.py                # ResNet-18 backbone and classification head
│   ├── train.py                # training entry point
│   └── predict.py                 # inference entry point
└── requirements.txt
```

## Evaluation Metrics

All experiments are evaluated on three metrics:

- **Balanced Accuracy** — mean per-class accuracy on the test split
- **AUROC** — area under the ROC curve for OOD detection (higher is better)
- **FPR95** — false positive rate when 95% of OOD samples are detected 
  (lower is better)

## Results

| Method | Bal. Acc | AUROC (Cars) | AUROC (DTD) | AUROC (Flowers102) |
|--------|----------|--------------|-------------|-------------------|
| E1: Baseline Softmax | 0.934 | 0.482 | 0.775 | 0.415 |
| E2: Independent BCE | 0.942 | 0.751 | 0.844 | 0.794 |
| E3: OOD Class | 0.944 | 1.000 | 1.000 | 0.999 |
| E4: Outlier Exposure | 0.936 | 0.873 | 0.933 | 0.826 |
| E5a: Energy (Post-Hoc) | 0.934 | 0.516 | 0.777 | 0.404 |
| E5b: Energy Fine-Tuning | 0.923 | 0.938 | 0.942 | 0.952 |
| E6: WOODS | 0.916 | 0.777 | 0.874 | 0.762 |

*E3 serves as an approximate upper bound given its access to labeled OOD data 
at training time.*

## Experiment Tracking

All experiments are tracked via Weights & Biases. The full project workspace 
is available [here](https://wandb.ai/deveshshah-university-of-michigan/OOD_Exploration).

## Citation

If you find this work useful, please cite:

```bibtex
@article{shah2024ood,
  title={Beyond Toy Benchmarks: Evaluating OOD Detection Methods on a 
         Real-World Plant Pathology Dataset},
  author={Shah, Devesh},
  year={2024}
}
```

## References

- Hendrycks & Gimpel (2017) — A Baseline for Detecting Misclassified and 
  Out-of-Distribution Examples
- Hendrycks et al. (2019) — Deep Anomaly Detection with Outlier Exposure
- Liu et al. (2020) — Energy-Based Out-of-Distribution Detection
- Katz-Samuels et al. (2022) — Training OOD Detectors in their Natural Habitats
