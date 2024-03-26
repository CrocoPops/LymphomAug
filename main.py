import torch
import os
import optuna

from tqdm import tqdm
from torchvision import transforms, datasets
from MyNets import AlexNet
from torch.utils.tensorboard import SummaryWriter

# Creating TensorBoard writer
writer = SummaryWriter()

# CONSTANTS
BASE_FOLDER = os.getcwd()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_FOLDER = os.path.join(os.getcwd(), "augmented")
LEARNING_RATES = [0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
BATCH_SIZES = [32, 64, 128, 256]
EPOCHS = [20, 30, 40, 50]


# TRANSFORMATION PIPELINE
# It is used to match the AlexNet input layer
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# LOADING THE DATA
# Using Pytorch feature that allows us to load images, divided in class, directly from the folders
dataset = datasets.ImageFolder(DATA_FOLDER, transform=transform)
# Splitting the data
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])

# DEFINING OBJECTIVE FUNCTION
# Will be used by Optuna to find the best hyperparams
def objective(trial):
    # DEFINING TRIAL VARIABLES
    lr = trial.suggest_categorical('lr', LEARNING_RATES)
    batch_size = trial.suggest_categorical('batch_size', BATCH_SIZES)
    num_epochs = trial.suggest_categorical("epochs", EPOCHS)

    # Creating Pytorch data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Creating the model
    model = AlexNet().to(DEVICE)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    # TRAINING LOOP
    bar1 = tqdm(range(num_epochs))
    bar1.set_description(f"Trial {trial.number}")
    for epoch in bar1:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        writer.add_scalar(f"Trial {trial.number} accuracy (train)", correct/total, epoch)

        # Evaluating the network
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs, labels) in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        writer.add_scalar(f"Trial {trial.number} accuracy (validation)", correct/total, epoch)

        # Check if the current trial meets the pruning criteria of Optuna
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return val_accuracy

# MAIN FUNCTION
if __name__ == "__main__":

    # Creating Optuna study to maximize the model accuracy
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=500)

    # Print the best hyperparams that Optuna found
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
