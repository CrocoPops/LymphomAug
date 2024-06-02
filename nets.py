
import torch.nn as nn
from torchvision import models

class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        
        # Pretrained AlexNet
        self.model = models.alexnet(weights = None, num_classes = 3)

        # Changing the last classification layer
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)