import torch
import os

from torchvision.transforms import v2

ORIGINAL_DATA = os.path.join(os.getcwd(), "data")
NUM_CLASSES = len(os.listdir(ORIGINAL_DATA))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ITERATIONS = 5
