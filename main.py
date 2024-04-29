import torch

from nets import AlexNet
from functions import get_data, train, test
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from constants import ITERATIONS, DEVICE
from augmentations import WaveletTransform, RGBRotation, HSVRotation, HSVSwap, Posterization, ColorWarping, ChromaticAberration, ColorQuantization

augmentations = {
    # "NoDA": [ v2.ToImage() ],

    # "Geometric" : [
    #     v2.RandomRotation(degrees=45),
    #     v2.RandomHorizontalFlip(),
    #     v2.RandomVerticalFlip(),
    #     v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15),
   
    # ],

    # "RGBRotation" : [  RGBRotation() ],

    # "HSVRotation" : [ HSVRotation() ],

    # "HSVSwap" : [ HSVSwap() ],

    # "Posterization" : [ Posterization(levels= 16) ],

    # "ColorWarping" : [ ColorWarping(frequency=5, amplitude=20) ],
    # "ChromaticAberration" : [ ChromaticAberration(shift_amount=5) ],
    # "ColorQuantization" : [ ColorQuantization(num_colors=8) ]

    # "Watershed" : [
    #     Watershed()
    # ]

    "QCE" : [WaveletTransform()]

}

for label, trans in augmentations.items():

    results = []

    for i in range(ITERATIONS):

        print("")
        print(f"[{label}] Starting iteration {i + 1}")

        print(f"[{label}] Started getting data..")
        train_dl, val_dl, test_dl = get_data(trans)

        print(f"[{label}] Started training..")
        model = AlexNet().to(DEVICE)
        train(model, train_dl, val_dl)

        print(f"[{label}] tarted testing..")
        accuracy = round(test(model, test_dl), 2)
        print(f"[{label}] Finished testing with accuracy: {accuracy}")

        results.append(accuracy)
    # Plot the results
    plt.plot(results, label=label)
    
plt.xlabel('Iteration')
plt.ylabel('Test accuracy')
plt.title('Test accuracies of different DA techniques')
plt.legend()
plt.show()