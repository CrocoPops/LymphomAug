import albumentations
import os
import numpy as np
import random
from tqdm import tqdm
import cv2
from PIL import Image

# CONSTANTS
BASE_FOLDER = os.getcwd()
IMAGE_FOLDERS = [
    os.path.join(BASE_FOLDER, "data", "0"), # IMAGE_FOLDERS[0] --> Image folder of class 0
    os.path.join(BASE_FOLDER, "data", "1"), # IMAGE_FOLDERS[1] --> Image folder of class 1
    os.path.join(BASE_FOLDER, "data", "2"), # IMAGE_FOLDERS[2] --> Image folder of class 2
]
AUGMENTATIONS_PER_CLASS = 1000

class RGBRotation(albumentations.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(RGBRotation, self).__init__(always_apply, p)

    def apply(self, image, **params):
        image_array = np.array(image)
        r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

        # rotate the RGB channels around each pixel in a clockwise direction
        rotated_r = np.roll(r, shift=1, axis=0) # rotate the red channel downwards by one pixel
        rotated_g = np.roll(g, shift=1, axis=1) # rotate the green channel downwards by one pixel
        rotated_b = np.roll(b, shift=-1, axis=0) # rotate the blue channel upwards by one pixel

        # merge the rotated RGB channels back into an image
        rotated_image_array = np.dstack((rotated_r, rotated_g, rotated_b))

        return Image.fromarray(rotated_image_array.astype('uint8'), 'RGB')

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
    RGBRotation(p=0.5)
])

# DELETING OLD AUGMENTED IMAGES
for img_folder in IMAGE_FOLDERS:
    for file in os.listdir(img_folder):
        if file.startswith("aug"):
            os.remove(os.path.join(img_folder, file))


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
