import os
import shutil
import albumentations
import random
import cv2
import numpy 
import pywt

from PIL import Image
from constants import *
from torch.utils.data import random_split

def setup_folders():
    if os.path.exists(TEMP_DATA):
        shutil.rmtree(TEMP_DATA)

    for folder in SUBFOLDERS:
        for i in range(NUM_CLASSES):  
            os.makedirs(os.path.join(TEMP_DATA, folder, str(i)))

def split_original_data():
    for i in range(NUM_CLASSES):
        filenames = os.listdir(os.path.join(ORIGINAL_DATA, str(i)))

        train, val, test = random_split(filenames, [0.7, 0.15, 0.15])

        for img in train:
            shutil.copy(os.path.join(ORIGINAL_DATA, str(i), img), os.path.join(TEMP_DATA, "train", str(i), img))
        for img in val:
            shutil.copy(os.path.join(ORIGINAL_DATA, str(i), img), os.path.join(TEMP_DATA, "validation", str(i), img))
        for img in test:
            shutil.copy(os.path.join(ORIGINAL_DATA, str(i), img), os.path.join(TEMP_DATA, "test", str(i), img))

def init_temp_data():
    setup_folders()
    split_original_data()

def select_random_file(folder_path):
    files = os.listdir(folder_path)
    random_file = random.choice(files)
    return os.path.join(folder_path, random_file)

def spatial_augment_class(label):
    for i in range(AUGMENTATIONS_PER_CLASS):
        img_filename = select_random_file(os.path.join(ORIGINAL_DATA, str(label)))
        image = cv2.imread(img_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = AUGMENTATIONS(image=image)["image"]
        cv2.imwrite(os.path.join(TEMP_DATA, "train", str(label), f"aug_{i}.png"), transformed)

def spatial_augment_all():
    for i in range(NUM_CLASSES):
        spatial_augment_class(i)

def fuse_images(label, filenames):
    images = [Image.open(os.path.join(TEMP_DATA, "train", str(label), fn)) for fn in filenames]

    images = [i.resize((224, 224)) for i in images]

    num_images = len(images)

    # Convert the images to numpy arrays
    images_array = [numpy.array(img) for img in images]

    # Separate the RGB channels of the images
    # [[R,G,B], [R,G,B], [R,G,B], [R,G,B]]
    channels = [[img_arr[:,:,i] for i in range(3)] for img_arr in images_array]

    # Apply the wavelet transform to each color channel
    fused_channels = []
    for i in range(3):
        # [RC1, RC2, RC3, RC4]
        coeffs = [pywt.dwt2(c[i], 'haar') for c in channels]
        
        cA = cV = cH = cD = 0
        for coeff in coeffs:
            cA += coeff[0] 
            cV += coeff[1][0]
            cH += coeff[1][1] 
            cD += coeff[1][2]

        cA = cA / num_images
        cV = cV / num_images
        cH = cH / num_images
        cD = cD / num_images
        

        fused_coeffs = (cA, (cV, cH, cD))
            
        # Reconstruct the fused channel
        fused_channel_array = pywt.idwt2(fused_coeffs, 'haar')
        
        # Clip values to ensure they are within 0-255 range
        fused_channel_array = (fused_channel_array * 255 / numpy.max(fused_channel_array)).astype('uint8')
        
        # Append the fused channel
        fused_channels.append(fused_channel_array)

    # Combine the fused color channels into an RGB image
    fused_image_array = numpy.stack(fused_channels, axis=-1)

    # Convert array back to uint8 and create PIL image
    return Image.fromarray(numpy.uint8(fused_image_array))

def dwt_augment_class(label):
    files = os.listdir(os.path.join(TEMP_DATA, "train", str(label)))
    # Generating new images
    for i in range(AUGMENTATIONS_PER_CLASS):
        c = numpy.random.choice(files, IMG_TO_FUSE)
        c_clean = [fn.split(".")[0] for fn in c]
        fn = "aug_" + "-".join(c_clean) + ".png"
        fuse_images(label, c).save(os.path.join(TEMP_DATA, "train", str(label), fn))

def dwt_augment_all():
    for i in range(NUM_CLASSES):
        dwt_augment_class(i)

def delete_augmented():
    for i in range(3):
        for fn in os.listdir(os.path.join(TEMP_DATA, "train", str(i))):
                if fn.startswith("aug_"):
                    os.remove(os.path.join(TEMP_DATA, "train", str(i), fn))