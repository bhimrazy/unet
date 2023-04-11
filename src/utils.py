"""Utils module.

This module contains utility functions for the segmentation task.

(c) 2023 Bhimraj Yadav. All rights reserved.
"""

import matplotlib.pyplot as plt

def show_images(images, masks, nmax=4, figsize=(16, 8)):
    """Display a list of images and masks in a uniform grid.

    Args:
        images (List[Image]): A list of images.
        masks (List[Image]): A list of masks.
        nmax (int, optional): The maximum number of images or masks to display. Defaults to 4.
        figsize (tuple, optional): The size of the figure. Defaults to (16, 8).
    """
    # Limit the number of images to display
    n = min(nmax, len(images))
    # Create a figure to plot the images
    fig, (ax1, ax2) = plt.subplots(2, n, figsize=figsize)

    # Loop through each image
    for i, ax in enumerate(ax1):
        image = images[i]

        ax.imshow(image[0], cmap='gray')
        ax.set_title(f"Image {i}")

    # Loop through each mask
    for i, ax in enumerate(ax2):
        mask = masks[i]

        ax.imshow(mask[0], cmap='gray')
        ax.set_title(f"Mask {i}")
