#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module with classes and functions related to data processed and loading"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "08/22"

import os
import cv2

import albumentations as A
from PIL import Image
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch import from_numpy
import yaml

# Load config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

class WF_Dataset(Dataset):
    def __init__(self, root_dir):
        super(WF_Dataset, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)
        files = os.listdir(root_dir)
        self.data += list(zip(files, [1] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, _ = self.data[index]
        root_and_dir = self.root_dir

        # Read image
        image = np.array(cv2.imread(os.path.join(root_and_dir, img_file), cv2.IMREAD_GRAYSCALE))

        # Apply transform
        image = resize(image=image)["image"]

        return image

class ProcessedDataset:
    def __init__(self, image, labels):
        self.image = image
        self.labels = labels

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        return self.image[idx], self.labels[idx] 


def load_prepare_dataset(root_dir):
    """
    Load Ux and Uy slices stored as grayscale images. Merge them into one ndarray and extract the inflow velocity for each image.
    """
    # Load dataset
    dataset_ux = WF_Dataset(root_dir=os.path.join(root_dir, 'ux')) # Load Ux
    dataset_uy = WF_Dataset(root_dir=os.path.join(root_dir, 'uy')) # Load Uy
    


    # Define arrays to store inflow velocity and images
    inflow = np.zeros((len(dataset_ux), dataset_ux[0].shape[0], 2))
    images = np.zeros((len(dataset_ux), dataset_ux[0].shape[0], dataset_ux[0].shape[1], 2))
    # Instantiate MinMaxScaler
    scaler = MinMaxScaler()

    # Iterate over all samples
    for c, (im_ux, im_uy) in enumerate(zip(dataset_ux, dataset_uy)):

        # Normalize all flow parameters and flow field using MinMaxScalers
        # Fit the scaler
        if c == 0:
            scaler.fit(im_ux.reshape(-1,1))
        # Rescale the images
        im_ux_tmp = scaler.transform(im_ux.reshape(-1, 1)); im_ux = im_ux_tmp.reshape(im_ux.shape)
        im_uy_tmp = scaler.transform(im_uy.reshape(-1, 1)); im_uy = im_uy_tmp.reshape(im_uy.shape)
        
        # Get inflow velocity for Ux and Uy and stack them
        inflow[c, :, :] = np.stack((im_ux[:, 0], im_uy[:, 0]), axis=-1)

        # Merge Ux and Uy images
        images[c, :, :, :] = np.stack((im_ux, im_uy), axis=-1)


    # Transforms array to tensors (adjust the shape to: NCHW)
    inflow = from_numpy(np.moveaxis(inflow, -1, 1))
    images = from_numpy(np.moveaxis(images, -1, 1))

    return images, inflow

# Resize transform
resize = A.Compose(
    [
        A.Resize(width=config['data']['final_size'][0], height=config['data']['final_size'][1], interpolation=Image.BICUBIC),
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        # A.pytorch.ToTensorV2(),
    ]
)