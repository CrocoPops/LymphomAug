import torch
import pywt

from nets import AlexNet
from functions import get_data, train, test
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from functions import *

from constants import DEVICE
from augmentations import NoneAug, RGBRotation, HSVRotation, DWTAverageFusion, DWTRandomFusion, DWTMaxFusion, DWTMinFusion

augmentations = [
    NoneAug,
    RGBRotation,
    HSVRotation,
    DWTAverageFusion,
    DWTRandomFusion,
    DWTMaxFusion,
    DWTMinFusion
]

for augmentation in augmentations:

    augmentation

    print(f"[{augmentation.__name__}] Started augmenting the images..")
    init_data()
    augment_data(augmentation())


    print(f"[{augmentation.__name__}] Started getting data..")
    train_dl, val_dl, test_dl = get_data()

    print(f"[{augmentation.__name__}] Started training..")
    model = AlexNet().to(DEVICE)
    results = train(model, train_dl, val_dl)

    print(f"[{augmentation.__name__}] tarted testing..")
    accuracy = round(test(model, test_dl), 2)
    print(f"[{augmentation.__name__}] Finished testing with accuracy: {accuracy}")

    # Plot the results
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(augmentation.__name__)

    ax1.set_ylabel('Accuracy')
    ax1.plot(results['train']['accuracy'], label='Train accuracy')
    ax1.plot(results['validation']['accuracy'], label='Validation accuracy')
    ax1.legend()

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.plot(results['train']['loss'], label='Train loss')
    ax2.plot(results['validation']['loss'], label='Validation loss')
    ax2.legend()
        
    plt.xlabel('Epoch')

    os.makedirs(os.path.join(os.getcwd(), "results", augmentation.__name__), exist_ok=True)

    plt.savefig(os.path.join(os.getcwd(), "results", augmentation.__name__, "plot"))

    with open(os.path.join(os.getcwd(), "results", augmentation.__name__, "test_accuracy.txt"), "w+") as file:
        file.write(str(accuracy))