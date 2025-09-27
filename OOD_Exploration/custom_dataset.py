import torch
import os

import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2


class PlantPathologyDataset(Dataset):
    def __init__(self, stage, dataset_path="./dataset/"):
        """
        Initializes the Plant Pathology Dataset.

        Args:
            stage (str): Subset of the dataset to use ('train', 'val', 'test').
            base_img_dir (str): Directory where the images and metadata are stored.
        """
        self.base_img_dir = os.path.join(dataset_path, "images_resized")
        self.stage = stage

        self.data = pd.read_csv(f"{dataset_path}/dataset.csv")
        self.data = self.data[self.data["stage"] == self.stage]

        self.label_encoding = {
            "healthy": 0,
            "scab": 1,
            "rust": 2,
            "frog_eye_leaf_spot": 3,
            "powdery_mildew": 4,
        }
        self.label_decoding = {v: k for k, v in self.label_encoding.items()}

        if self.stage == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20),
                    v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    v2.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )

        print(f"{stage} dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        img_path = f"{self.base_img_dir}/{item['image_path']}"
        image = Image.open(img_path)

        label = item["label"]
        label = self.label_encoding[label]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}


if __name__ == "__main__":
    stage = "train"
    data_set = PlantPathologyDataset(stage=stage)
    data_loader = DataLoader(data_set, batch_size=9, shuffle=False)
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    batch = next(iter(data_loader))
    images_batch = batch["image"]
    labels_batch = batch["label"]
    for i in range(len(images_batch)):
        row, col = i // 3, i % 3
        ax[row, col].imshow(images_batch[i].permute(1, 2, 0))
        ax[row, col].set_title(data_set.label_decoding[labels_batch[i].item()])
        ax[row, col].axis("off")
    fig.tight_layout()
    # plt.show()

    idx = 0
    first_item = data_set[idx]
    print(first_item["image"].shape, first_item["label"])

    print()
    print("Dataset Loaded Successfully")
