import torch
import os

from torchvision.transforms import v2

# FOLDERS 
ORIGINAL_DATA = os.path.join(os.getcwd(), "data")
FOLDERS = ['train', 'test']
AUGMENTED_FOLDER = os.path.join(os.getcwd(), "augmented_data")

# USEFUL
NUM_CLASSES = len(os.listdir(ORIGINAL_DATA))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ITERATIONS = 5
RANDOM_STATE = 22

# DATA AUG
AUGS_PER_CLASS = 1000

# LEARNING
EPOCHS = 20
BATCH_SIZE = 8
LR = 0.01
WD = 0.0001
MOMENTUM = 0.7

# TORCH TRANSFORM (NOT USED IN DA)
TRANSFORM = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])