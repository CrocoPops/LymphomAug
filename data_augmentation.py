import albumentations
import os
import random
from tqdm import tqdm
import cv2

# CONSTANTS
BASE_FOLDER = os.getcwd()
IMAGE_FOLDERS = [
    os.path.join(BASE_FOLDER, "augmented", "0"), # IMAGE_FOLDERS[0] --> Image folder of class 0
    os.path.join(BASE_FOLDER, "augmented", "1"), # IMAGE_FOLDERS[1] --> Image folder of class 1
    os.path.join(BASE_FOLDER, "augmented", "2"), # IMAGE_FOLDERS[2] --> Image folder of class 2
]
AUGMENTATIONS_PER_CLASS = 1000

# FUNCTIONS
def select_random_file(folder_path):
    files = os.listdir(folder_path)
    random_file = random.choice(files)
    return os.path.join(folder_path, random_file)

# AUGMENTATION PIPELINE
augmentations = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.RandomRotate90(p=0.5),
    albumentations.RandomBrightnessContrast(p=0.5),
    albumentations.ShiftScaleRotate(p=0.5),
])

# AUGMENTING IMAGES
for img_folder in IMAGE_FOLDERS:
    bar1 = tqdm(range(AUGMENTATIONS_PER_CLASS))
    bar1.set_description(f"Class {IMAGE_FOLDERS.index(img_folder)}")
    for i in bar1:
        img_filename = select_random_file(img_folder)
        image = cv2.imread(img_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = augmentations(image=image)["image"]
        cv2.imwrite(os.path.join(img_folder, f"aug_{i}.png"), image)
