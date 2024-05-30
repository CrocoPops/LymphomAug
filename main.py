import torch
import pywt

from nets import AlexNet
from functions import get_data, train, test
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from functions import *

from constants import DEVICE
from augmentations import *

augmentations = [
    NoneAug(),
    RandomGeometricTransform(),
    RGBRotation(),
    HSVRotation(),
    DWTAverageFusion(),
    DWTRandomFusion(),
    DWTMaxFusion(),
    DWTMinFusion(),
    SaltAndPepper(prob = 0.01),
    ShuffleSquares(square_size=25),
]

accuracies = {}

for augmentation in augmentations:

    # Augmenting the images
    print(f"[{augmentation.__class__.__name__}] Started augmenting the images..")
    init_data()
    augment_data(augmentation)

    # Splitting the dataset into train, validation and test set
    print(f"[{augmentation.__class__.__name__}] Splitting data..")
    train_dl, val_dl, test_dl = get_data()

    # Training the model
    print(f"[{augmentation.__class__.__name__}] Started training..")
    model = AlexNet().to(DEVICE)
    results = train(model, train_dl, val_dl)

    # Testing the model and printing the test result
    print(f"[{augmentation.__class__.__name__}] Started testing..")
    accuracy = round(test(model, test_dl), 2)
    print(f"[{augmentation.__class__.__name__}] Finished testing with accuracy: {accuracy}")
    print("")

    # Plot the results
    save_plot(
        augmentation.__class__.__name__,
        results['train']['accuracy'],
        results['validation']['accuracy'],
        results['train']['loss'],
        results['validation']['loss'],
        os.path.join(os.getcwd(), "results", augmentation.__class__.__name__)       
    )

    # Save the results in the dict (will be saved at the end)
    accuracies[augmentation.__class__.__name__] = accuracy


# Save accuracies on file
with open(os.path.join(os.getcwd(), "results", "combined_results.txt"), "w+") as file:
        for k,v in accuracies.items():
             file.write(f'{k}: {v}\n')