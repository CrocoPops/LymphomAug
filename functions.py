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
    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

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
