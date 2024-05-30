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

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


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
        val_class_dir = os.path.join(os.path.join(AUGMENTED_FOLDER, "validation"), class_folder)
        test_class_dir = os.path.join(os.path.join(AUGMENTED_FOLDER, "test"), class_folder)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        images = os.listdir(class_path)
        numpy.random.shuffle(images)

        num_images = len(images)
        train_split = int(0.8 * num_images)
        val_split = int(0.1 * num_images)

        train_images = images[:train_split]
        val_images = images[train_split:train_split + val_split]
        test_images = images[train_split + val_split:]

        # Copy images to respective directories
        for img_name in train_images:
            shutil.copy(os.path.join(class_path, img_name), os.path.join(train_class_dir, img_name))
        for img_name in val_images:
            shutil.copy(os.path.join(class_path, img_name), os.path.join(val_class_dir, img_name))
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
    val_ds = ImageFolder(os.path.join(AUGMENTED_FOLDER, 'validation'), TRANSFORM)
    test_ds = ImageFolder(os.path.join(AUGMENTED_FOLDER, 'test'), TRANSFORM)

    train_dl = DataLoader(train_ds, BATCH_SIZE, True)
    val_dl = DataLoader(val_ds, BATCH_SIZE, False)
    test_dl = DataLoader(test_ds, BATCH_SIZE, False)

    return train_dl, val_dl, test_dl

def train(model, train_dl, val_dl):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    best_accuracy = 0.0
    best_model_params = None

    train_acc = []
    train_loss = []

    val_acc = []
    val_loss = []

    bar = tqdm(range(EPOCHS), ncols=100, unit="epoch", leave=False, desc=f"Epoch 0")
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
        train_acc.append(accuracy)
        train_loss.append(running_loss)


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
                running_loss += loss.item() * inputs.size(0)

        accuracy = correct / total

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_params = model.state_dict()

        
        val_acc.append(accuracy)
        val_loss.append(running_loss)

        bar.set_postfix({'running_loss': running_loss, 'val_acc': accuracy})

    model = model.load_state_dict(best_model_params)
    return {'train': {'accuracy' : train_acc, 'loss' : train_loss}, 'validation': {'accuracy' : val_acc, 'loss' : val_loss}}

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

def save_plot(augmentation_name, train_acc, val_acc, train_loss, val_loss, path):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(augmentation_name)

    ax1.set_ylabel('Accuracy')
    ax1.plot(train_acc, label='Train accuracy')
    ax1.plot(val_acc, label='Validation accuracy')
    ax1.legend()

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.plot(train_loss, label='Train loss')
    ax2.plot(val_loss, label='Validation loss')
    ax2.legend()
        
    plt.xlabel('Epoch')

    plt.savefig(path)