import torch
from tqdm import tqdm
from constants import DEVICE
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from constants import ORIGINAL_DATA

def get_data(transform):

    if not transform:
        transform = [v2.Resize((224, 224))]

    full_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.Compose(transform),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(ORIGINAL_DATA, full_transform)
    train_ds, val_ds, test_ds = random_split(dataset, [0.7, 0.15, 0.15])

    train_dl = DataLoader(train_ds, 64, True)
    val_dl = DataLoader(val_ds, 64, False)
    test_dl = DataLoader(test_ds, 64, False)

    return train_dl, val_dl, test_dl

def train(model, train_dl, val_dl):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    bar = tqdm(range(50), ncols=100, unit="epoch", leave=False, desc=f"Epoch 0")
    for epoch in bar:
        bar.set_description(f"Epoch {epoch}")
        running_loss = 0.0
        correct, total = 0, 0
        model.train()
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total


        # Validation of the model.
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        bar.set_postfix({'running_loss': running_loss, 'val_acc': accuracy})

def test(model, test_dl):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
