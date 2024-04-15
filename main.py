import torch
import os

from tqdm import tqdm
from MyNets import AlexNet, DenseNet, ResNet
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from constants import *
from functions import *

import matplotlib.pyplot as plt

# TRAINING LOOP
def train(model_class, da):

    model = model_class().to(DEVICE) 

    train_dataset = ImageFolder(os.path.join(TEMP_DATA, "train"), transform=TRANSFORMS)
    val_dataset = ImageFolder(os.path.join(TEMP_DATA, "validation"), transform=TRANSFORMS)

    # Creating Pytorch data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)

    os.makedirs(os.path.join(BASE_FOLDER, "saves"), exist_ok=True)

    best_validation_accuracy = 0.0
    train_losses = []
    validation_losses = []
    print(f"Training:")
    for epoch in range(model.epochs):

        # Training the network  
        model.train() 
        
        # (Re-)Init stats counters
        running_loss, total, correct = 0.0, 0, 0

        for inputs, labels in train_loader:
            
            # Forward prop and back prop
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Update the stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = correct / total
        train_losses.append(running_loss)

        # Evaluating the network
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs) 
                loss = criterion(outputs, labels)
   
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        validation_losses.append(running_loss)

        print(f"({epoch}) Running loss: {running_loss:.4}, Train accuracy: {train_accuracy:.4}, Validation accuracy: {val_accuracy:.4} ")

        if val_accuracy > best_validation_accuracy:
            torch.save(model.state_dict(), os.path.join(BASE_FOLDER, "saves", f"{model.__class__.__name__}-{da}.pt"))
            best_validation_accuracy = val_accuracy

    return train_losses, validation_losses  

def test_model(model_class, da):
    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(BASE_FOLDER, "saves", f"{model.__class__.__name__}-{da}.pt")))
    test_dataset = ImageFolder(os.path.join(TEMP_DATA, "test"), transform=TRANSFORMS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def plot_losses(train_losses, val_losses): 
    plt.clf()    
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend()
    plt.show()

# NO DATA AUGMENTATION
init_temp_data()
train_acc, val_acc = train(AlexNet, "None")
print(f"Best trained AlexNet test accuracy: {test_model(AlexNet, "None")}")

# DWT AUGMENTATION
dwt_augment_all()
train_acc, val_acc = train(AlexNet, "DWT")
print(f"Best trained AlexNet test accuracy: {test_model(AlexNet, "DWT")}")

# ROTATIONS, FLIP, etc
delete_augmented()
spatial_augment_all()
train_acc, val_acc = train(AlexNet, "Spatial")
print(f"Best trained AlexNet test accuracy: {test_model(AlexNet, "Spatial")}")

# COMBO
dwt_augment_all()
train_acc, val_acc = train(AlexNet, "Spatial")
print(f"Best trained AlexNet test accuracy: {test_model(AlexNet, "Spatial")}")
