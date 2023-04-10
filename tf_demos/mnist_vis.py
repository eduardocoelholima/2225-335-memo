################################################################
# mnist_vis.py 
#
# Routines to view MNIST data
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################

import matplotlib.pyplot as plt
import numpy as np
import math

# Constants
MNIST_DIM=28
def view_digit( i, image_tensor, labels ):
    reshaped = image_tensor[i].reshape([MNIST_DIM, MNIST_DIM])
    plt.imshow( reshaped, cmap='gray_r' )
    plt.gca().set_title('Label: {}'.format( np.argmax(labels[i])))
    plt.show()

def view_array( image_tensor, labels, cols ):
    # Adapted from: https://medium.com/the-data-science-publication/how-to-plot-mnist-digits-using-matplotlib-65a2e0cc068
    ( images, vlength ) = image_tensor.shape

    # Hack -- require two rows, show empty values if necessary
    rows = max(2, math.ceil(images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 2 * rows ))
    for i in range(images):
        ax = axes[i//cols, i%cols]
        ax.imshow( image_tensor[i].reshape(MNIST_DIM,MNIST_DIM), cmap='gray')
        ax.set_title('Label: {}'.format( np.argmax(labels[i])))
        #ax.set_title('Label: {}'.format(labels[i]))
    plt.tight_layout()
    plt.show()
