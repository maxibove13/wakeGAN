#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to train the CGAN proposed by Zhang & Zhao"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

# Built-in modules
import os
import time

# Third party modules
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import yaml

# Local modules
from data.utils import load_prepare_dataset, ProcessedDataset
from models.model_01 import Generator, Discriminator, Embedding, initialize_weights

root_dir = os.path.join('data', 'preprocessed', 'train')

# Load config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

CHANNELS = 2

def train():

    print("cDCGAN training:")

    # Set device
    device = torch.device(config['train']['device']) if torch.cuda.is_available() else 'cpu'
    print(
        f"...\n"
        f"Using device: {torch.cuda.get_device_name()}"
        f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB")

    # Load and process images, extract inflow velocity 
    images, inflow = load_prepare_dataset(root_dir)
    print(
        f"...\n"
        f"Loading data with {images.shape[0]} samples\n"
        f"Flow parameters shape: {inflow.shape}\n" # (N,2,64)
        f"Flow field data shape: {images.shape}\n" # (N,2,64,64)
        )
    # Load image dataset in the pytorch manner
    dataset = ProcessedDataset(images, inflow)

    # Initialize models and send them to device
    print('Initializing Embedding, Generator and Discriminator...')
    # Embedding takes (C, H)
    emb = Embedding(CHANNELS, images.shape[-1]).to(device)
    # Generator takes (C, H, f_g)
    gen = Generator(CHANNELS, config['data']['final_size'][0], config['model']['f_g']).to(device)
    # Discriminator takes (C) (4 channels, Ux, Uy, Ux_in, Uy_in)
    disc = Discriminator(CHANNELS+2, config['model']['f_d']).to(device)

    print('Defining losses and optimizers...')
    # Define losses
    criterion = torch.nn.BCELoss()
    # Define optimizer
    opt_gen = torch.optim.Adam(gen.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))

    # Define kfold and splits
    n_splits = config['train']['n_splits']
    kfold = KFold(n_splits=n_splits, shuffle=True)
    # Iterate over kfolds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(images)):


        # Load flow field data
        train_subsampler = SubsetRandomSampler(train_idx)
        trainloader = DataLoader(
            dataset, 
            batch_size=config['train']['batch_size'], 
            pin_memory=True, 
            num_workers=config['train']['num_workers'], 
            sampler=train_subsampler)

        test_subsampler = SubsetRandomSampler(train_idx)
        valloader = DataLoader(
            dataset, 
            batch_size=config['train']['batch_size'], 
            pin_memory=True, 
            num_workers=config['train']['num_workers'], 
            sampler=test_subsampler)

        # Training loop
        print(f"Starting training: \n")
        print(
            f"  k-fold: {fold+1}/{n_splits}\n"
            f"  Training samples on this fold: {len(train_subsampler)}\n"
            f"  Testing samples on this fold: {len(test_subsampler)}\n"
            f"  Number of epochs: {config['train']['num_epochs']}\n"
            f"  Mini batch size: {config['train']['batch_size']}\n"
            f"  Number of batches: {len(trainloader)}\n"
            f"  Learning rate: {config['train']['lr']}\n"
        )

    # Iterate over epochs 
    for epoch in range(config['train']['num_epochs']):
        # Iterate over mini batches
        for idx, (image, label) in enumerate(trainloader):

            image = image.float().to(device)
            label = label.float().to(device)

            # Initialize weights of gen
            initialize_weights(gen)
            initialize_weights(disc)
            # Generate flow field prediction (fake image) with training data in mini batch
            fake = gen(label) # gen generates flow field image from label inflow velocity

            ## Embed labels with images
            real_emb = torch.cat((emb(label), image), 1)
            fake_emb = torch.cat((emb(label), fake), 1)
            ## Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            opt_disc.zero_grad() # Reset gradients to zero
            pred_real = disc(real_emb) # Evaluate real data 
            loss_real = criterion(pred_real, torch.ones_like(pred_real)) # Calculate loss
            pred_fake = disc(fake_emb) # Evaluate fake data 
            loss_fake = criterion(pred_real, torch.zeros_like(pred_fake)) # Calculate loss
            loss_d = loss_fake + loss_real # Combine losses
            loss_d.backward() # Backward pass
            opt_disc.step() # Update weights

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            opt_gen.zero_grad() # Reset gradients to zero
            loss_g = criterion(pred_fake, torch.ones_like(pred_fake)) # Calculate loss
            loss_g.backward() # Backward pass
            opt_gen.step() # Update weights

            print(fake.shape)

        # Print progress every epoch
        print(
            f"Epoch [{epoch+1}/{config['train']['num_epochs']} - "
            f"Loss D_real: {loss_real:.4f}, Loss D_fake: {loss_fake:.4f}, Loss G: {loss_g:.4f}]"
            )
            # Train discriminator on real [U, \mu] and generated [U_gen, \mu] data by feeding 

            # Train generator on discriminator output

            # Generate flow field prediction with validation data in mini batch

            # Test model on validation data (generate flow field prediction with testing data)

        # Save model on each epoch

        # Compute accuracy like metrics on each epoch

    return

if __name__ == "__main__":
    tic = time.time()
    train()
    toc = time.time()

    print(f"Training duration: {((toc-tic)/60):.2f} m ")