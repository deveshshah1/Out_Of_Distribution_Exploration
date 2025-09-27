import torch
from tqdm import tqdm
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os


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


def predict():
    model = resnet50()
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 5)
    model.load_state_dict(torch.load('trained_results/best_model_weights.pth', map_location=torch.device("cpu"), weights_only=True))
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # all_test_images = os.listdir("dataset/OOD_images")
    # all_test_images = [img for img in all_test_images if img.endswith(('.jpeg'))]
    # all_test_images.sort()

    # for path in all_test_images:
    #     print(path)
    #     test_image = Image.open(f"dataset/OOD_images/{path}")
    #     test_image = transform(test_image).unsqueeze(0).to(device)

    #     with torch.no_grad():
    #         output = model(test_image)
    #         _, predicted = torch.max(output, 1)
    #         predicted_label = list(label_encoding.keys())[predicted.item()]
    #         softmax_output = torch.softmax(output, dim=1)
    #         print(f"Max softmax output: {softmax_output.max().item()}")
    #         # print(f'Predicted model softmax output: {torch.softmax(output, dim=1)}')

    test_dataset = PlantPathologyDataset(subset='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_softmax = []
    all_labels = []

    for images, labels in tqdm(test_loader):
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            softmax_output = torch.softmax(outputs, dim=1)
            all_softmax.append(softmax_output.cpu())
            all_labels.append(labels)
    
    df = pd.DataFrame({"softmax": torch.cat(all_softmax).numpy(), "labels": torch.cat(all_labels).numpy()})
    df.to_csv("predictions.csv", index=False)


if __name__=="__main__":
    predict()
