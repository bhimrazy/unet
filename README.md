# [WIP] PyTorchUNet: An Efficient Implementation of UNet Architecture from Scratch Using PyTorch
PyTorchUNet is a PyTorch-based implementation of the UNet architecture for semantic image segmentation. 
This repository contains a comprehensive implementation of the UNet architecture, including both the encoder and decoder modules, using PyTorch 2.0. 
The code is easy to understand and can be easily extended to different datasets and problems.

## Installation
To use PyTorchUNet, you need to have Python 3.8 or higher installed on your system. You can install the required Python packages by running the following command:
```shell
pip install -r requirements.txt
```
This will install all the required packages, including PyTorch and its dependencies.

## Usage
To train and test the PyTorchUNet model, you need to prepare your data in the appropriate format. The input images and maks should be in separate folders, and each image should have a corresponding masklabel with the same name. The folder structure should look like this:

```
data/
    images/
        image1.png
        image2.png
        ...
    masks/
        mask1.png
        mask2.png
        
```
To prepare dataset, run the following command:
```shell
python data_setup.py
```

To train the model, run the following command:

```shell
python train.py 
```
This will train the PyTorchUNet model on the specified dataset and save the checkpoints at the specified path.

To test the model, run the following command:
```shell
# python test.py 
```
This will load the specified checkpoint and test the model on the specified dataset.

## Test Preview
![image](https://user-images.githubusercontent.com/46085301/231221990-17de0bb8-5aca-4f59-b457-a2771185c16f.png)

## Contributing
If you find a bug or have a feature request, please open an issue or submit a pull request. We welcome contributions from the community!

## Acknowledgments
This implementation is based on the original UNet paper by Ronneberger et al. [1]. I would like to thank the PyTorch team for providing an excellent deep learning framework.

## References
[1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2015.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
