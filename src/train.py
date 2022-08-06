#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to train the CGAN proposed by Zhang & Zhao"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import os

# Third party modules

# Local modules

def train():

    # Set device

    # Define train dataset

    # Define validation dataset

    # Transform data by using MinMaxScalers

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