import torch
from tqdm import tqdm
import copy
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
label_encoding = {"healthy": 0, "scab": 1, "rust": 2, "frog_eye_leaf_spot": 3, "powdery_mildew": 4}


class PlantPathologyDataset(Dataset):
    def __init__(self, subset, transform=None):
        data = pd.read_csv(f"dataset/{subset}.csv")
        self.filepaths = data["image"].values
        self.target = data["labels"].values
        self.transform = transform

    def __len__(self):
        return 100
        # return len(self.filepaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = f"dataset/images/{self.filepaths[idx]}"
        image = Image.open(img_path)

        label = self.target[idx]
        label = label_encoding[label]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    print(f'Using device: {device}')

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 5)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model = train_model(model, criterion, optimizer, scheduler, num_epochs=2)
    torch.save(best_model.state_dict(), 'best_model_weights.pth')


def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor()
    ])

    train_dataset = PlantPathologyDataset("train", transform=transform)
    val_dataset = PlantPathologyDataset("val", transform=transform)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=32, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=32, shuffle=False)
    }

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
    
    return best_model


if __name__ == '__main__':
    main()