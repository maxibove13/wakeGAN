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
import numpy as np
from matplotlib import cm, pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.utils import save_image
import yaml

# Local modules
from data.utils import load_prepare_dataset, ProcessedDataset
from models.model_01 import Generator, Discriminator, Embedding, initialize_weights
from visualization.utils import plot_metrics, calc_mse

root_dir = os.path.join('data', 'preprocessed', 'train')

# Load config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

CHANNELS = config['data']['channels']
NUM_EPOCHS = config['train']['num_epochs']
CLIM_UX = config['data']['figures']['clim_ux']
CLIM_UY = config['data']['figures']['clim_uy']

def train():

    print("cDCGAN training:")

    # Set device
    device = torch.device(config['train']['device']) if torch.cuda.is_available() else 'cpu'
    print(
        f"\n"
        f"Using device: {torch.cuda.get_device_name()}"
        f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB")

    # Load and process images, extract inflow velocity 
    images, inflow = load_prepare_dataset(root_dir)
    n_test_data = len(os.listdir(os.path.join('data', 'preprocessed', 'test', 'ux')))
    print(
        f"\n"
        f"Loading training and validation data with {images.shape[0]} samples ({n_test_data} samples remain for testing, {n_test_data/images.shape[0]*100:.1f}%)\n"
        f"Flow parameters shape (N,C,H): {inflow.shape}\n" # (N,2,64)
        f"Flow field data shape (N,C,H,W): {images.shape}\n" # (N,2,64,64)
        )

    # Load image dataset in the pytorch manner
    dataset = ProcessedDataset(images, inflow)

    # Initialize models and send them to device
    print('Initializing Embedding, Generator and Discriminator')
    # Embedding takes (C, H)
    emb = Embedding(CHANNELS, images.shape[-1]).to(device)
    # Generator takes (C, H, f_g)
    gen = Generator(CHANNELS, config['data']['final_size'][0], config['model']['f_g']).to(device)
    # Discriminator takes (C) (4 channels, Ux, Uy, Ux_in, Uy_in)
    disc = Discriminator(CHANNELS+2, config['model']['f_d']).to(device)

    # Define losses
    criterion = torch.nn.BCELoss()
    print(f'Using {criterion} loss')
    # Define optimizer
    opt_gen = torch.optim.Adam(gen.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
    print(f'Defining Adam optimizers for both Gen and Disc')

    gen.train()
    disc.train()

    # Define kfold and splits
    n_splits = config['train']['n_splits']
    kfold = KFold(n_splits=n_splits, shuffle=True)

    # Iterate over kfolds
    rmse_folds = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(images)):


        # Load flow field data
        train_subsampler = SubsetRandomSampler(train_idx)
        trainloader = DataLoader(
            dataset, 
            batch_size=config['train']['batch_size'], 
            pin_memory=True, 
            num_workers=config['train']['num_workers'], 
            sampler=train_subsampler)

        val_subsampler = SubsetRandomSampler(val_idx)
        valloader = DataLoader(
            dataset, 
            batch_size=config['train']['batch_size'], 
            pin_memory=True,
            num_workers=config['train']['num_workers'], 
            sampler=val_subsampler)

        initialize_weights(gen)
        initialize_weights(disc)

        # Training loop
        print(f"\n")
        print(f"Starting training: \n")
        print(
            f"  k-fold: {fold+1}/{n_splits}\n"
            f"  Training samples on this fold: {len(train_subsampler)}\n"
            f"  Validation samples on this fold: {len(val_subsampler)}\n"
            f"  Number of epochs: {NUM_EPOCHS}\n"
            f"  Mini batch size: {config['train']['batch_size']}\n"
            f"  Number of batches: {len(trainloader)}\n"
            f"  Learning rate: {config['train']['lr']}\n"
        )

        # Iterate over epochs 
        for epoch in range(NUM_EPOCHS):
            # Iterate over mini batches
            for idx, (image, label) in enumerate(trainloader):

                image = image.float().to(device)
                label = label.float().to(device)

                # noise = torch.randn((config['train']['batch_size'], 64, 8, 8)).to(device)

                # Initialize weights of gen
                torch.autograd.set_detect_anomaly(False)
                # Generate flow field prediction (fake image) with training data in mini batch
                fake = gen(label) # gen generates flow field image from label inflow velocity
                ## Embed labels with images
                real_emb = torch.cat((image, emb(label)), 1)
                fake_emb = torch.cat((fake, emb(label)), 1)
                ## Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                pred_real = disc(real_emb) # Evaluate real data 
                loss_real = criterion(pred_real.reshape(-1), torch.ones_like(pred_real).reshape(-1)) # Calculate loss
                pred_fake = disc(fake_emb) # Evaluate fake data 
                loss_fake = criterion(pred_fake.reshape(-1), torch.zeros_like(pred_fake).reshape(-1)) # Calculate loss
                loss_d = (loss_fake + loss_real) / 2 # Combine losses
                opt_disc.zero_grad() # Reset gradients to zero
                loss_d.backward(retain_graph=True) # Backward pass
                opt_disc.step() # Update weights

                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                loss_g = criterion(disc(fake_emb).reshape(-1), torch.ones_like(disc(fake_emb)).reshape(-1)) # Calculate loss
                opt_gen.zero_grad() # Reset gradients to zero
                loss_g.backward() # Backward pass
                opt_gen.step() # Update weights


            # Test model on validation data (generate flow field prediction with validation data)
            gen.eval()
            with torch.no_grad():
                rmse = [0, 0]
                # Iterate over minibatches of val data
                print()
                for idx, (images, labels) in enumerate(valloader):
                    # Generate images for this minibatch
                    images = images.float().to(device)
                    labels = labels.float().to(device)
                    ims_gen = gen(labels)
                    # Iterate over all images in this batch
                    for i, (image, im_gen) in enumerate(zip(images, ims_gen)):
                        rmse[0] += calc_mse(image[0], im_gen[0])
                        rmse[1] += calc_mse(image[1], im_gen[1])
                    rmse[0] /= len(images)
                    rmse[1] /= len(images)
                rmse[0] /= len(valloader)
                rmse[1] /= len(valloader)
                rmse[0] = torch.sqrt(rmse[0]).cpu().detach().numpy()
                rmse[1] = torch.sqrt(rmse[1]).cpu().detach().numpy()



                # Plot figure with images real and fake
                grid_ims = [image[0], im_gen[0], im_gen[0]-image[0],image[1], im_gen[1], im_gen[1]-image[1]]
                if epoch == 0:
                    fig_im = plt.figure(dpi=300)
                    grid = ImageGrid(fig_im, 111, nrows_ncols=(2,3), axes_pad=0.1, share_all='True', cbar_location='right', cbar_mode='each')
                for c, (ax, cax, im) in enumerate(zip(grid, grid.cbar_axes, grid_ims)):
                    

                    if c < 3:
                        im = im * (CLIM_UX[1]-CLIM_UX[0]) + CLIM_UX[0]
                        ax.set_title('Ux')
                        vmin = CLIM_UX[0]; vmax = CLIM_UX[1]
                    else:
                        im = im * (CLIM_UY[1]-CLIM_UY[0]) + CLIM_UY[0]
                        ax.set_title('Uy')
                        vmin = CLIM_UY[0]; vmax = CLIM_UY[1]

                    
                    im = ax.imshow(
                        im.cpu(),
                        cmap=cm.coolwarm, 
                        interpolation='none',
                        vmin=vmin, vmax=vmax)

                    cb = cax.colorbar(im)
                    # if epoch == 0:
                    #     if c == 0:
                    #         ax.set_ylabel("$Ux$")
                    #     elif c == 3:
                    #         ax.set_ylabel("$Uy$")
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # fig_im.colorbar(im, cax=cax, orientation='vertical')


                fig_im.suptitle(f"RMSE_Ux: {torch.sqrt(calc_mse(image[0], im_gen[0])):.3f}, RMSE_Uy: {torch.sqrt(calc_mse(image[1], im_gen[1])):.3f}")
                fig_im.savefig(os.path.join('figures', 'image_comparison.png'), dpi=300)
                
                        
            # Print progress every epoch
            print(
                f"Epoch [{epoch+1:03d}/{NUM_EPOCHS:03d} - "
                f"Loss D_real: {loss_real:.3f}, Loss D_fake: {loss_fake:.3f}"
                f", Loss G: {loss_g:.3f}, RMSE Val.: ({rmse[0]:.3f}, {rmse[1]:.3f})]"
                )
            # Plot loss
            if epoch == 0:
                # Initialize figure for loss and rmse evolution
                fig, axs = plt.subplots(2,1,dpi=300)
                # Initialize list containing metrics
                loss_disc = []; loss_gen = []
                rmse_evol_ux = []; rmse_evol_uy = []
            # Plot metrics
            plot_metrics(loss_disc, loss_gen, loss_d, loss_g, epoch, fig, axs, NUM_EPOCHS, rmse, rmse_evol_ux, rmse_evol_uy)

            

        # Append RMSE to list of RMSE per fold
        rmse_folds.append(rmse)
        # Average RMSE of all folds
        print(f"Average RMSE of fold {fold+1}/{n_splits}: {rmse_folds[fold]}")
        gen.train()

    print(f"Average RMSE of all folds: {sum(rmse_folds)/len(rmse_folds):.2f}")

            # Train discriminator on real [U, \mu] and generated [U_gen, \mu] data by feeding 

            # Train generator on discriminator output

            # Generate flow field prediction with validation data in mini batch


        # Save model on each epoch

        # Compute accuracy like metrics on each epoch

    return

if __name__ == "__main__":
    tic = time.time()
    train()
    toc = time.time()

    print(f"Training duration: {((toc-tic)/60):.2f} m ")