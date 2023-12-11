"""
Configuration file for the project.

This file contains all the configuration variables for the project.

Attributes:
    DATA_DIR (str): The path to the data directory.
    MODEL_DIR (str): The path to the model directory.
    LOG_DIR (str): The path to the log directory.
    DEVICE (str): The device to use for training.
    BATCH_SIZE (int): The batch size for training.
    NUM_WORKERS (int): The number of workers for the data loader.
    NUM_EPOCHS (int): The number of epochs for training.
    LEARNING_RATE (float): The learning rate for training.
    WEIGHT_DECAY (float): The weight decay for training.
    IMG_SIZE (int): The size of the image.
    NUM_CLASSES (int): The number of classes.
    CHANNELS (List[int]): A list of channels for convolutionals block.
    OUT_CHANNELS (int): The number of output channels.
    MODEL_NAME (str): The name of the model.
    MODEL_PATH (str): The path to the model.
    LOG_PATH (str): The path to the log file.

(c) 2023 Bhimraj Yadav. All rights reserved.
"""
import os
import torch
import datetime

# Directories
DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "logs"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
BATCH_SIZE = 2
NUM_WORKERS = 4
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Image size
IMG_SIZE = 256

# Number of classes
NUM_CLASSES = 1

# Model parameters
CHANNELS = [1, 64, 128, 256, 512, 1024]
OUT_CHANNELS = 1

# Model name
now = datetime.datetime.now()
MODEL_NAME = f"unet_{now.strftime('%Y-%m-%d_%H-%M-%S')}"

# Model path
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pth")

# Log path
LOG_PATH = os.path.join(LOG_DIR, f"{MODEL_NAME}.log")
