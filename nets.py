import torchvision.models as models
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=3, epochs=20):
        super(AlexNet, self).__init__()
        self.epochs = epochs

        # Pretrained AlexNet
        self.model = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)

        # Changing the last classification layer
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)