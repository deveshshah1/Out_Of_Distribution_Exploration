import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from custom_dataset import PlantPathologyDataset
from model import BaselineModel
from .utils.custom_scheduluer import NoamScheduler


# Global config
with open("./configs/config_training.yaml", "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}


class PyLDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path

    def setup(self, stage=None):
        self.train_set = PlantPathologyDataset(
            stage="train", dataset_path=self.dataset_path
        )
        self.val_set = PlantPathologyDataset(
            stage="val", dataset_path=self.dataset_path
        )

    def train_dataloader(self):
        sampler = WeightedRandomSampler(
            weights=self.train_set.get_class_weights(),
            num_samples=len(self.train_set),
            replacement=True,
        )

        return DataLoader(
            self.train_set,
            batch_size=config_training["training_hyperparameters"]["batch_size"],
            sampler=sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=config_training["training_hyperparameters"]["batch_size"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=4,
            persistent_workers=True,
        )


class PyLModel(pl.LightningModule):
    def __init__(self, wandb_logger=None):
        super().__init__()
        self.save_hyperparameters()
        self.wandb_logger = wandb_logger

        self.LABEL_ENCODING = config_training["plant_label_encoding"]
        self.LABEL_DECODING = {v: k for k, v in self.LABEL_ENCODING.items()}

        self.model = BaselineModel(num_classes=len(self.LABEL_ENCODING))
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["label"]

        _, outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels).float() / len(labels)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/accuracy", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        labels = batch["label"]

        _, outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels).float() / len(labels)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        # Get inputs
        inputs = batch["image"]
        labels = batch["label"]
        ids = batch["id"]

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            emb, logits = self.model(inputs)

        outputs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(outputs, dim=1)

        true_label_name = [self.LABEL_DECODING[label.item()] for label in labels]
        pred_label_name = [self.LABEL_DECODING[pred.item()] for pred in preds]

        return {
            "id": ids,
            "predicted_label": pred_label_name,
            "true_label": true_label_name,
            "embedding": emb,
            "outputs": outputs,
            "logits": logits,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config_training["training_hyperparameters"]["learning_rate"],
            weight_decay=config_training["training_hyperparameters"]["weight_decay"],
        )
        scheduler = NoamScheduler(
            optimizer,
            warmup_steps=config_training["training_hyperparameters"]["warmup_steps"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    dataset = PyLDataModule(dataset_path="./dataset/")
    model = PyLModel()
    breakpoint()
