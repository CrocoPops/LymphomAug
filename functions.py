from ctypes.wintypes import DOUBLE
import os
import shutil
import numpy
import random

from PIL import Image
from torch.utils.data import random_split
from constants import *
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from numpy.random import RandomState
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from ml_things import plot_confusion_matrix


import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


random.seed(RANDOM_STATE)

def create_folders():

    if os.path.isdir(AUGMENTED_FOLDER):
        shutil.rmtree(AUGMENTED_FOLDER)

    for folder in FOLDERS:
        for i in range(NUM_CLASSES):
            os.makedirs(os.path.join(AUGMENTED_FOLDER, folder, str(i)))

def split_data():
    for class_folder in os.listdir(ORIGINAL_DATA):

        class_path = os.path.join(ORIGINAL_DATA, class_folder)
        train_class_dir = os.path.join(os.path.join(AUGMENTED_FOLDER, "train"), class_folder)
        test_class_dir = os.path.join(os.path.join(AUGMENTED_FOLDER, "test"), class_folder)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        images = os.listdir(class_path)
        RandomState(RANDOM_STATE).shuffle(images)

        num_images = len(images)
        train_split = int(0.8 * num_images)
        test_split = int(0.2 * num_images) # line not necessary

        train_images = images[:train_split]
        test_images = images[train_split:]

        # Copy images to respective directories
        for img_name in train_images:
            shutil.copy(os.path.join(class_path, img_name), os.path.join(train_class_dir, img_name))
        for img_name in test_images:
            shutil.copy(os.path.join(class_path, img_name), os.path.join(test_class_dir, img_name))

def select_random_file(folder_path):
    filenames = [f for f in os.listdir(folder_path) if not f.startswith("aug")]
    return random.choice(filenames)

def augment_data(augmentation):
    if augmentation.__class__.__name__ != "NoneAug":
        for label in range(NUM_CLASSES):
            print(f"[{augmentation.__class__.__name__}] Augmenting class '{label}'")
            bar = tqdm(range(AUGS_PER_CLASS), ncols=100, unit="image", leave=False, desc=f"Image 0")
            
            for i in bar:    
                bar.set_description(f"Image {i}")
                images = []
                for _ in range(augmentation.args_images):
                    image_filename = select_random_file(os.path.join(AUGMENTED_FOLDER, 'train', str(label)))
                    images.append(Image.open(os.path.join(AUGMENTED_FOLDER, 'train', str(label), image_filename)))
                
                image = augmentation(images)
                image.save(os.path.join(AUGMENTED_FOLDER, 'train', str(label), f"aug_{i}.png"))
    else:
        print(f"[{augmentation.__class__.__name__}] Using original dataset, no augmentation needed.")

def init_data():
    create_folders()
    split_data()

def get_data():

    train_ds = ImageFolder(os.path.join(AUGMENTED_FOLDER, 'train'), TRANSFORM)
    test_ds = ImageFolder(os.path.join(AUGMENTED_FOLDER, 'test'), TRANSFORM)

    train_dl = DataLoader(train_ds, BATCH_SIZE, True)
    test_dl = DataLoader(test_ds, BATCH_SIZE, False)

    return train_dl, test_dl

def train(model, train_dl):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
    # optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=EPOCHS)
    best_model_params = None


    bar = tqdm(range(EPOCHS), ncols=100, unit="epoch", leave=False, desc=f"Epoch 0")
    for epoch in bar:
        bar.set_description(f"Epoch {epoch}")
        running_loss_train = 0.0
        correct, total = 0, 0
        model.train()
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy_train = correct / total
        scheduler.step()            

        bar.set_postfix({'running_loss': running_loss_train, 'train_acc': accuracy_train, 'lr': optimizer.param_groups[0]['lr']})

        # Save the training permonance for statistical purposes
        train_acc = accuracy_train
        
        # Save the best model of the last epoch
        best_model_params = model.state_dict()
    

    model = model.load_state_dict(best_model_params)

    # Return the training and validation accuracy of the best model in this iteration of training
    return {'accuracy' : train_acc}

def test(model, test_dl):
    model.eval()
    correct, total = 0, 0
    labels_forMetrics = []
    predictions_forMetrics = []

    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            labels_forMetrics.append(labels.cpu().numpy()) # Move the tensor to the cpu and convert it to numpy
            predictions_forMetrics.append(predicted.cpu().numpy()) # Move the tensor to the cpu and convert it to numpy

        metrics = Metrics(numpy.concatenate(labels_forMetrics), numpy.concatenate(predictions_forMetrics))

    return metrics

def save_plot(augmentation_name, train_acc, test_acc, path):
    plt.clf()
    fig, ax = plt.subplots()
    fig.suptitle(augmentation_name)

    ax.set_ylabel('Accuracy')
    ax.plot(range(1, len(train_acc)+1), train_acc, label='Train accuracy')
    ax.plot(range(1, len(test_acc)+1), test_acc, label='Test accuracy')
    ax.legend()

    plt.xlabel('Iteration')

    plt.savefig(path)

class Metrics:
    def __init__(self, labels, predictions):
        self.labels = labels
        self.predictions = predictions

    def get_classification_report(self, output_dict=False):
        return classification_report(y_true=self.labels, y_pred=self.predictions, target_names=['class 0', 'class 1', 'class 2'], output_dict=output_dict, zero_division=0)
    
    def get_accuracy(self):
        return accuracy_score(self.labels, self.predictions)
    
    def get_confusion_matrix(self):
        return confusion_matrix(self.labels, self.predictions, labels=[0, 1, 2], normalize=None)

    def get_confusion_matrix(self):
        return confusion_matrix(self.labels, self.predictions, normalize='all', labels=[0, 1, 2])
    
    def save_classification_report(self, path):
        with open(path, 'w') as f:
            f.write(self.get_classification_report(output_dict=False))

    def save_confusion_matrix(self, path, augmentation_name):
        # Get the confusion matrix
        cm = self.get_confusion_matrix()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=['class 0', 'class 1', 'class 2'], yticklabels=['class 0', 'class 1', 'class 2'], ax=ax)

        # Set axis labels and title
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f"Confusion Matrix - {augmentation_name}")

        # Save the plot
        plt.savefig(path)
        plt.close()