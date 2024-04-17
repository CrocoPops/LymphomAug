from torchvision.transforms import v2
import os
import torch
import albumentations

# Training device used (cpu if available, cpu otherwise)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Folder constants
ORIGINAL_DATA = os.path.join(os.getcwd(), "data")
TEMP_DATA = os.path.join(os.getcwd(), "temp_data")
TRAIN_FOLDER = os.path.join(TEMP_DATA, "train")
VAL_FOLDER = os.path.join(TEMP_DATA, "val")
TEST_FOLDER = os.path.join(TEMP_DATA, "test")
SUBFOLDERS = [TRAIN_FOLDER, VAL_FOLDER, TEST_FOLDER]

# Number of classes in the dataset
NUM_CLASSES = len(os.listdir(ORIGINAL_DATA))

# Original data split percentage
TRAIN_PERC = 0.8
VAL_PERC = 0.1
TEST_PERC = 0.1

# Constants used in augmentations
AUGMENTATIONS_PER_CLASS = 500
IMG_TO_FUSE = 2

# Transformation pipelines
BASE_TRANSFORM = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

AUGMENTATIONS = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.RandomRotate90(p=0.5),
    albumentations.RandomBrightnessContrast(p=0.5),
    albumentations.ShiftScaleRotate(p=0.5),
])

