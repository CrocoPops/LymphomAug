import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from torch.utils.data import DataLoader
from nets import AlexNet

from functions import *
import optuna
from constants import *

def objective(trial):


    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 64, 256])
    epochs = trial.suggest_categorical('epochs', [10, 20, 50, 100])

    print("    ", "-" * 5)
    print("")
    print(f"TRIAL NUMBER {trial.number}")
    print("")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay:  {weight_decay}")
    print(f"Batch size:    {batch_size}")
    print(f"Epochs:        {epochs}")

    print("")

    train_dl = get_dataloader(TRAIN_FOLDER, BASE_TRANSFORM, batch_size)
    val_dl = get_dataloader(VAL_FOLDER, BASE_TRANSFORM, batch_size, False)

    model = AlexNet().to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    bar = tqdm(range(epochs), ncols=100, unit="epoch", leave=False, desc=f"Epoch 0")
    for epoch in bar:
        bar.set_description(f"Epoch {epoch}")
        running_loss = 0.0
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        bar.set_postfix({'running_loss': running_loss})

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
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    print("")
    print("    ", "-" * 5)
    print("")

    return accuracy

os.makedirs(os.path.join(os.getcwd(), "databases"), exist_ok=True)

studies = ["none", "spat", "fuse", "combo"]

for s in studies:
    init_temp_data()

    if s == "spat":
        spatial_augment_all()
    elif s == "fuse":
        dwt_augment_all()
    elif s == "combo":
        spatial_augment_all()
        dwt_augment_all()

    study = optuna.create_study(
            storage=f"sqlite:///databases/{s}.sqlite3",  
            study_name=s,
            direction="maximize"
        )
    # Start the optimization process
    study.optimize(objective, n_trials=30)
