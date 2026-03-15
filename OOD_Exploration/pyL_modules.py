import yaml
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from torchmetrics import Accuracy
from pytorch_lightning.utilities import CombinedLoader
from sklearn.metrics import roc_auc_score

from custom_dataset import PlantPathologyDataset
from model import BaselineModel
from utils.custom_scheduluer import NoamScheduler


# Global config
with open("./configs/config_training.yaml", "r") as file:
    config_training = yaml.safe_load(file)
    config_training = {k: v["value"] for k, v in config_training.items()}


class PyLDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.base_dataset_path = config_training["dataset_configs"]["base_dataset_path"]
        self.dataset_name = config_training["dataset_configs"]["plantpathology"]
        self.ood_dataset_name = config_training["dataset_configs"]["imagenet-o"]
        self.batch_size = config_training["training_hyperparameters"]["batch_size"]

    def setup(self, stage=None):
        self.train_set_id = PlantPathologyDataset(
            stage="train", base_dataset_path=self.base_dataset_path, dataset_name=self.dataset_name
        )
        self.train_set_ood = PlantPathologyDataset(
            stage="train", base_dataset_path=self.base_dataset_path, dataset_name=self.ood_dataset_name
        )
        self.val_set_id = PlantPathologyDataset(
            stage="val", base_dataset_path=self.base_dataset_path, dataset_name=self.dataset_name
        )
        self.val_set_ood = PlantPathologyDataset(
            stage="val", base_dataset_path=self.base_dataset_path, dataset_name=self.ood_dataset_name
        )

    def train_dataloader(self):
        sampler = WeightedRandomSampler(
            weights=self.train_set_id.get_class_weights(),
            num_samples=len(self.train_set_id),
            replacement=True,
        )

        train_id_loader = DataLoader(
            self.train_set_id,
            batch_size=self.batch_size,
            sampler=sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
            persistent_workers=True,
        )
        train_ood_loader = DataLoader(
            self.train_set_ood,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
            persistent_workers=True,
        )

        combined_loader = CombinedLoader(
            {"id": train_id_loader, "ood": train_ood_loader}, mode="max_size_cycle"
        )
        return combined_loader
    
    def val_dataloader(self):
        val_id_loader = DataLoader(
            self.val_set_id,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=4,
            persistent_workers=True,
        )
        val_ood_loader = DataLoader(
            self.val_set_ood,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=4,
            persistent_workers=True,
        )

        return [val_id_loader, val_ood_loader]

class PyLModel(pl.LightningModule):
    def __init__(self, wandb_logger=None):
        super().__init__()
        self.save_hyperparameters()
        self.wandb_logger = wandb_logger

        self.LABEL_ENCODING = config_training["plant_label_encoding"]
        self.LABEL_DECODING = {v: k for k, v in self.LABEL_ENCODING.items()}

        num_classes = len(self.LABEL_ENCODING)
        self.model = BaselineModel(num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lambda_ood = config_training["training_hyperparameters"]["lambda_ood"]
        self.warmup_epochs = config_training["training_hyperparameters"]["warmup_epochs"]

        self.val_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self.val_balanced_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_per_class_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )

        self.val_id_ood_scores = []
        self.val_ood_ood_scores = []

    def training_step(self, batch, batch_idx):
        batch_id = batch["id"]
        batch_ood = batch["ood"]

        inputs_id = batch_id["image"]
        labels_id = batch_id["label"]

        _, outputs_id = self.model(inputs_id)
        loss_id = self.criterion(outputs_id, labels_id)

        inputs_ood = batch_ood["image"]
        _, outputs_ood = self.model(inputs_ood)
        uniform = torch.ones_like(outputs_ood) / outputs_ood.size(1)
        loss_ood = F.cross_entropy(outputs_ood, uniform)

        effective_lambda = 0.0 if self.current_epoch < self.warmup_epochs else self.lambda_ood
        loss = loss_id + effective_lambda * loss_ood

        _, preds = torch.max(outputs_id, 1)
        acc = torch.sum(preds == labels_id).float() / len(labels_id)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_id", loss_id, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_ood", loss_ood, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/accuracy", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch["image"]
        labels = batch["label"]

        _, outputs = self.model(inputs)
        ood_scores = -outputs.softmax(dim=-1).max(dim=-1).values

        if dataloader_idx == 0:
            loss = self.criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            self.val_acc.update(preds, labels)
            self.val_balanced_acc.update(preds, labels)
            self.val_per_class_acc.update(preds, labels)
            self.val_id_ood_scores.append(ood_scores.detach().cpu())
            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
        elif dataloader_idx == 1:
            self.val_ood_ood_scores.append(ood_scores.detach().cpu())

    def on_validation_epoch_end(self):
        # Accuracy metrics
        val_acc = self.val_acc.compute()
        balanced_acc = self.val_balanced_acc.compute()
        per_class_acc = self.val_per_class_acc.compute()

        self.log("val/accuracy", val_acc, prog_bar=True)
        self.log("val/balanced_accuracy", balanced_acc, prog_bar=True)

        for class_idx, acc in enumerate(per_class_acc):
            class_name = self.LABEL_DECODING[class_idx]
            self.log(f"val/accuracy_{class_name}", acc)

        # Reset metrics for next epoch
        self.val_acc.reset()
        self.val_balanced_acc.reset()
        self.val_per_class_acc.reset()

        # ODO metrics
        id_scores = torch.cat(self.val_id_ood_scores)    # low scores, ID images
        ood_scores = torch.cat(self.val_ood_ood_scores) # high scores, OOD images

        labels = torch.cat([torch.zeros(len(id_scores)), torch.ones(len(ood_scores))])
        all_scores = torch.cat([id_scores, ood_scores])

        auroc = torch.tensor(roc_auc_score(labels.numpy(), all_scores.numpy()))
        threshold = torch.quantile(ood_scores, 0.05)
        fpr95 = torch.mean((id_scores > threshold).float()).item() * 100

        self.log("val/ood_auroc", auroc, prog_bar=True)
        self.log("val/ood_fpr95", fpr95, prog_bar=True)

        self.val_id_ood_scores.clear()
        self.val_ood_ood_scores.clear()

    def predict_step(self, batch, batch_idx):
        # Get inputs
        inputs = batch["image"]
        labels = batch["label"]
        ids = batch["id"]

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            emb, logits = self.model(inputs)

        outputs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(outputs, dim=-1)

        true_label_name = [
            self.LABEL_DECODING[label.item()]
            if label.item() in self.LABEL_DECODING
            else "Unknown"
            for label in labels
        ]
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
