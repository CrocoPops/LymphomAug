import os
import albumentations
import torch
import numpy

from PIL import Image
from torchvision import transforms

# GENERAL CONSTANTS
BASE_FOLDER = os.getcwd()
ORIGINAL_DATA = os.path.join(BASE_FOLDER, "data")
TEMP_DATA = os.path.join(BASE_FOLDER, "temp_data")
SUBFOLDERS = ["train", "validation", "test"]
NUM_CLASSES = 3

# AUGMENTATIONS CONSTANTS
IMG_TO_FUSE = 2
AUGMENTATIONS_PER_CLASS = 500
AUGMENTATIONS = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.RandomRotate90(p=0.5),
    albumentations.RandomBrightnessContrast(p=0.5),
    albumentations.ShiftScaleRotate(p=0.5),
])

# NETWORK CONSTANTS
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

