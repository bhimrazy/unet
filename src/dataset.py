"""Dataset module.

This module contains the dataset class for the segmentation task.

(c) 2023 Bhimraj Yadav. All rights reserved.
"""
import glob
import os

import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import BATCH_SIZE, NUM_WORKERS, ROOT_DIR

class SegmentationDataset(Dataset):
    """
    Segmentation dataset class.

    Args:
        images (List[str]): A list of paths to images.
        masks (List[str]): A list of paths to masks.
        transform (Callable): A function/transform that takes in an image and returns a transformed version.

    """

    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Get an item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The image and mask tensors.
        """
        # Load the image and mask
        image = Image.open(self.images[idx])
        mask = Image.open(self.masks[idx])

        # Apply transformations
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


transform = T.Compose(
    [
        T.ToImage(),
        T.Resize(256, antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.ToPureTensor(),
    ]
)

# load data files
images = sorted(glob.glob(os.path.join(ROOT_DIR, "data/images/*.jpg")))
masks = sorted(glob.glob(os.path.join(ROOT_DIR, "data/masks/*.jpg")))

# split the dataset
train_size = int(0.8 * len(images))
val_size = len(images) - train_size

# split the dataset without random

train_images, val_images = images[:train_size], images[train_size:]
train_masks, val_masks = masks[:train_size], masks[train_size:]

# create the datasets
train_dataset = SegmentationDataset(train_images, train_masks, transform=transform)
val_dataset = SegmentationDataset(val_images, val_masks, transform=transform)

# create the dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
