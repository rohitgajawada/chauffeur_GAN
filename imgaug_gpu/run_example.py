import torch
from torchvision import transforms
from PIL import Image
import imgaug as ia
import imgauggpu as iag

import matplotlib.pyplot as plt



def plot_images(image_reference, augmenter_cpu, augmenter_gpu):
    fig, ax_vector = plt.subplots(3, 3)

    ax_vector.plot()

    ax_vector[0, 1].imshow(image_reference)
    # Plotting cpu rows
    for i in range(3):
        ax_vector[1, i].imshow(augmenter_cpu(image_reference))
    # Plotting cpu rows
    for i in range(3):
        ax_vector[2, i].imshow(augmenter_gpu(image_reference))


    plt.axis('Off')
    plt.show()









if __name__ == '__main__':


    #Load an image
    rat_image = Image.open('quokka.jpg')

    # ADDING operations
    add_cpu = iag.AugmenterCPU([ia.augmenters.Add((-20, 20))])
    add_gpu = iag.Augmenter([iag.Add((-20, 20))])

    plot_images(rat_image, add_cpu, add_gpu)

    # MULTIPLYING operation.

    # Plot comparisons


    # Show a cool example the sometimes usage.

    sometimes_multiply = iag.Augmenter(iag.Sometimes(iag.Multiply(2.0)))

    #for i in 10_rats:
    #    sometimes_multiply(rat_image)

    #    plot_the_rat images

    print (" >>>>>>> RUN THE SPEED TEST TO CHECK THE ADVANTAGES <<<<<<<<<<")