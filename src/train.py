#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to train the CGAN proposed by Zhang & Zhao"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import os
import cv2

# Third party modules
import numpy as np
import torch
import yaml

# Local modules
from data.utils import load_prepare_dataset
from models.model_01 import Generator, Discriminator, Embedding

root_dir = os.path.join('data', 'preprocessed', 'train')

# Load config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)


def train():

    print("cDCGAN training:")

    # Set device
    device = torch.device(config['train']['device']) if torch.cuda.is_available() else 'cpu'
    print(
        f"...\n"
        f"Using device: {torch.cuda.get_device_name()}"
        f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB")

    # Load and prepare training dataset 
    # Normalize all images, isolate inflow velocity (flow parameter) from the images and merge Ux and Uy slice images into one array of shape (N, H, W, 2). 
    inflow, images = load_prepare_dataset(root_dir)
    print(
        f"...\n"
        f"Loading data with {images.shape[0]} samples\n"
        f"Flow parameters: {inflow.shape}\n"
        f"Flow field data: {images.shape}\n"
        )

    # Initialize models and send them to device
    print('Initializing Embedding, Generator and Discriminator...')
    # Embedding receives 2 channels and the Height of the image
    emb = Embedding(images.shape[-1], images.shape[1]).to(device)
    # Generator receives flow parameters
    gen = Generator().to(device)
    # Discriminator receives 4 channels (Ux, Uy, Ux_in, Uy_in)
    disc = Discriminator(images.shape[-1]+2).to(device)

    # Load model


    # Define optimizer

    # Define losses

    # Load pretrained model (if)

    # Iterate over epochs 

        # Iterate over mini batches

            # Generate flow field prediction with training data in mini batch ()

            # Train discriminator on real [U, \mu] and generated [U_gen, \mu] data by feeding 

            # Train generator on discriminator output

            # Generate flow field prediction with validation data in mini batch

            # Test model on validation data (generate flow field prediction with testing data)

        # Save model on each epoch

        # Compute accuracy like metrics on each epoch

    return

if __name__ == "__main__":
    train()