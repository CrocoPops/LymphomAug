import torch.nn as nn
import torch
from torchvision import models

# Define the neural network model (AlexNet)
class AlexNet(nn.Module):
    def __init__(self, num_classes=3, epochs=20):  # 3 classes in your case
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.epochs = epochs

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, num_classes=3, epochs=5):
        super(DenseNet, self).__init__()
        self.features = models.densenet121(weights = models.DenseNet121_Weights.DEFAULT).features
        self.classifier = nn.Linear(1024, num_classes)
        self.epochs = epochs

    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=3, epochs=5):
        super(ResNet, self).__init__()
        self.resnet = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.epochs = epochs

    def forward(self, x):
        return self.resnet(x)