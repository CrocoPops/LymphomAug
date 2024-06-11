import os
import scipy.io
from PIL import Image
 
# Loading the MAT file
mat = scipy.io.loadmat('DatasColor_29.mat')['DATA']

# Labels in the mat file are in the range 1-3, we need to subtract one to
# make them in range 0-2
labels = [x - 1 for x in mat[0][1][0]]
images = mat[0][0][0]

# Creating the folders
os.makedirs(os.path.join('data', '0'), exist_ok=True)
os.makedirs(os.path.join('data', '1'), exist_ok=True)
os.makedirs(os.path.join('data', '2'), exist_ok=True)

# Saving the images as PNG files
for index, (image_arr, label) in enumerate(zip(images, labels)):
    Image.fromarray(image_arr).save(os.path.join('data', str(label), f"{index}.png"))

