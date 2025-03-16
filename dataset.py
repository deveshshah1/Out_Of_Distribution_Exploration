import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


class PlantPathologyDataset(Dataset):
    def __init__(self, subset, base_img_dir="dataset/images"):
        """
        Initializes the Plant Pathology Dataset.
        
        Args:
            subset (str): Subset of the dataset to use ('train', 'val', 'test').
            base_img_dir (str): Directory where the images are stored.
        """
        self.base_img_dir = base_img_dir
        self.subset = subset

        data = pd.read_csv(f"dataset/{subset}.csv")
        self.filepaths = data["image"].values
        self.target = data["labels"].values

        self.label_encoding = {"healthy": 0, "scab": 1, "rust": 2, "frog_eye_leaf_spot": 3, "powdery_mildew": 4}
        self.label_decoding = {v: k for k, v in self.label_encoding.items()}

        if self.subset == "train":
            self.transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(20),
                                transforms.ToTensor()
                            ])
        else:
            self.transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                            ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = f"{self.base_img_dir}/{self.filepaths[idx]}"
        image = Image.open(img_path)

        label = self.target[idx]
        label = self.label_encoding[label]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label
        }
    

if __name__ == "__main__":
    subset = "train"
    data_set = PlantPathologyDataset(subset=subset)
    data_loader = DataLoader(data_set, batch_size=9, shuffle=False)
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    batch = next(iter(data_loader))
    images_batch = batch["image"]
    labels_batch = batch["label"]
    for i in range(len(images_batch)):
        row, col = i//3, i%3
        ax[row, col].imshow(images_batch[i].permute(1, 2, 0))
        ax[row, col].set_title(data_set.label_decoding[labels_batch[i].item()])
        ax[row, col].axis("off")
    fig.tight_layout()
    plt.show()