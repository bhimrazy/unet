#  argument parser for testing with model checkpoint, image path , save path
#  and other parameters
#  python test.py --checkpoint_path ./checkpoints/ --image_path ./test_images/ --save_path ./test_results/ --image_size 256 --batch_size 1 --num_workers 1 --gpu_ids 0

import argparse
import os
from PIL import Image
import numpy as np

import torch

from src.config import CHANNELS, DEVICE, OUT_CHANNELS
from src.model import UNet
from src.dataset import transform

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--checkpoint_path",
    type=str,
    help="Path to load model checkpoint",
    required=True,
)
parser.add_argument(
    "-i", "--image_path", type=str, help="Path to load image for testing", required=True
)
parser.add_argument(
    "-s",
    "--save_path",
    type=str,
    default="artifacts/test.png",
    help="Path to save image for testing",
)

args = parser.parse_args()

if __name__ == "__main__":
    # load model
    model = UNet(channels=CHANNELS, out_channels=OUT_CHANNELS)
    model.to(DEVICE)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint)

    # check if image with given path exists
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found at {args.image_path}")

    # load image
    image = Image.open(args.image_path)
    image = image.convert("L")
    
    # transform image
    image = transform(image)    

    # add batch dimension
    image = image.unsqueeze(0)

    # predict
    model.eval()
    with torch.no_grad():
        prediction = model(image.to(DEVICE))
        prediction = prediction.cpu().squeeze(0).squeeze(0)

    # save prediction
    prediction = torch.sigmoid(prediction)
    # prediction = (prediction > 0.5).float()
    prediction = prediction.numpy()
    prediction = np.uint8(prediction * 255)
    
    prediction = Image.fromarray(prediction)
    prediction.save(args.save_path, format="PNG")

    print(f"Saved prediction image to {args.save_path}")