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
    DWTAverageFusion(),
    RGBRotation(),
    HSVRotation(),
    HSVSwap(),
    SaltAndPepper(prob=0.01),
    ShuffleSquares(square_size=25),
    RandomGeometricTransform(),
    Rotation(),
    Flip(),
    GridColored(),
    RandomBrightness(),
    RandomShifts(),
    ComboGeometricBrightness(),
    ComboGeometricRGBRotation(),
    #ComboBrightnessRandomShifts(),
    ComboGeometricShift(),
    ComboGeometricHSVRotation(),
    #ComboHSVShift(),
    ComboGeometricGridColored(),
    ComboGeometricShuffleSquares(square_size=25)
]

accuracies = {}

# Create the results folder if it does not exist
if(not os.path.isdir(os.path.join(os.getcwd(), "results"))):
    os.makedirs(os.path.join(os.getcwd(), "results"))




for augmentation in augmentations:
    # Create the folders for metrics
    if(not os.path.isdir(os.path.join(os.getcwd(), "results", augmentation.__class__.__name__))):
        os.makedirs(os.path.join(os.getcwd(), "results", augmentation.__class__.__name__))

    # Augmenting the images
    print(f"[{augmentation.__class__.__name__}] Started augmenting the images..")
    init_data()
    augment_data(augmentation)

    # Splitting the dataset into train, validation and test set
    print(f"[{augmentation.__class__.__name__}] Splitting data..")
    train_dl, test_dl = get_data()

    # Run n iteration of training and testing and the compute the average accuracy
    results_iterations = {'train': [], 'test': []}
    accuracy_DA = []


    for i in range(ITERATIONS):
        print (f"[{augmentation.__class__.__name__}] Iteration {i+1}/{ITERATIONS}")
        # Training the model
        print(f"[{augmentation.__class__.__name__}] Started training..")
        model = AlexNet().to(DEVICE)

        results_train = train(model, train_dl)
        results_iterations['train'].append(results_train['accuracy'])

        # Testing the model and printing the test result
        print(f"[{augmentation.__class__.__name__}] Started testing..")

        metrics = test(model, test_dl)

        accuracy = round(metrics.get_accuracy(), 2)
        accuracy_DA.append(accuracy)
        results_iterations['test'].append(accuracy)

        print(f"[{augmentation.__class__.__name__}] Finished testing with metrics:")
        print()
        print(metrics.get_classification_report(output_dict=False))
        print("")

        # Save the matrix and the confusion matrix at last iteration
        if(i == ITERATIONS - 1):
            print(" Saving metrics..")
            metrics.save_classification_report(os.path.join(os.getcwd(), "results", augmentation.__class__.__name__, "classification_report.txt"))
           # metrics.plot_confusion_matrix(augmentation_name=augmentation.__class__.__class__.__name__)
            metrics.save_confusion_matrix(os.path.join(os.getcwd(), "results", augmentation.__class__.__name__, "confusion_matrix.png"), augmentation_name=augmentation.__class__.__name__)

    # Compute the average of each Data Augmentation
    accuracy_DA = round(sum(accuracy_DA) / len(accuracy_DA), 2)

    print()
    print(f"[{augmentation.__class__.__name__}] Average accuracy: {accuracy_DA}")

    # # Plot the results
    # save_plot(
    #     augmentation.__class__.__name__,
    #     results_iterations['train'],
    #     results_iterations['test'],
    #     os.path.join(os.getcwd(), "results", augmentation.__class__.__name__)       
    # )

    # Save the results in a dictionary
    accuracies[augmentation.__class__.__name__] = accuracy_DA


# Save accuracies on file
with open(os.path.join(os.getcwd(), "results", "average_accuracy_DATAAUG.txt"), "w+") as file:
    for k,v in accuracies.items():
        file.write(f'{k}: {v}\n')