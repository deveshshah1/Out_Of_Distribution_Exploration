import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BaselineModel(nn.Module):
    def __init__(self, num_classes=5):
        super(BaselineModel, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = BaselineModel()
    print(model)

    sample_input = torch.randn(1, 3, 224, 224)  # Example input tensor
    output = model(sample_input)
    print(output.shape)  # Should print torch.Size([1, 5])
