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
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import yaml

# Local modules
from data.utils import load_prepare_dataset, ProcessedDataset
from models.model_01 import Generator, Discriminator, Embedding, initialize_weights
from visualization.utils import plot_metrics, calc_mse, plot_flow_field_comparison

root_dir = os.path.join('data', 'preprocessed', 'train')
root_dir_test = os.path.join('data', 'preprocessed', 'test')

# Load config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

CHANNELS = config['data']['channels']
NUM_EPOCHS = config['train']['num_epochs']
CLIM_UX = config['data']['figures']['clim_ux']
CLIM_UY = config['data']['figures']['clim_uy']
KFOLD = config['validation']['kfold']
BATCH_SIZE = config['train']['batch_size']
MULTIBATCH = config['train']['multibatch']

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
    images_test, inflow_test = load_prepare_dataset(root_dir_test)
    n_test_data = len(os.listdir(os.path.join('data', 'preprocessed', 'test', 'ux')))
    print(
        f"\n"
        f"Loading training and validation data with {images.shape[0]} samples ({n_test_data} samples remain for testing, {n_test_data/images.shape[0]*100:.1f}%)\n"
        f"Flow parameters shape (N,C,H): {inflow.shape}\n" # (N,2,64)
        f"Flow field data shape (N,C,H,W): {images.shape}\n" # (N,2,64,64)
        )

    # Load image dataset in the pytorch manner
    dataset = ProcessedDataset(images, inflow)
    dataset_test = ProcessedDataset(images_test, inflow_test)

    # Initialize models and send them to device
    # Embedding takes (C, H)
    # emb = Embedding(CHANNELS, images.shape[-1]).to(device)
    # Generator takes (C, H, f_g)
    gen = Generator(CHANNELS, config['data']['final_size'][0], config['models']['f_g']).to(device)
    # Discriminator takes (C, f_d)
    disc = Discriminator(CHANNELS, config['models']['f_d'], config['data']['final_size'][1]).to(device)
    print(
        # f"Initializing Embedding with {sum(p.numel() for p in emb.parameters())} params\n"
        f"Initializing Generator with {sum(p.numel() for p in gen.parameters())} params\n"
        f"Initializing Discriminator with {sum(p.numel() for p in disc.parameters())} params\n"
        )

    # Define losses
    criterion = torch.nn.BCELoss()
    print(f'Using {criterion} loss')
    # Define optimizer
    opt_gen = torch.optim.Adam(gen.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))
    print(f'Defining Adam optimizers for both Gen and Disc\n')

    # Load models
    if config['models']['load']:
        print('Loading pretrained model')
        # Load Gen
        checkpoint_gen = torch.load(
            os.path.join('models', config['models']['name_gen']),
            map_location = device
        )
        gen.load_state_dict(checkpoint_gen["state_dict"])
        opt_gen.load_state_dict(checkpoint_gen["optimizer"])
        for param_group in opt_gen.param_groups:
            param_group["lr"] = config['train']['lr']
        # Load Disc
        checkpoint_disc = torch.load(
            os.path.join('models', config['models']['name_disc']),
            map_location = device
        )
        disc.load_state_dict(checkpoint_disc["state_dict"])
        opt_disc.load_state_dict(checkpoint_disc["optimizer"])
        for param_group in opt_disc.param_groups:
            param_group["lr"] = config['train']['lr']

    gen.train()
    disc.train()

    # Define kfold and splits
    n_splits = config['validation']['n_splits']
    kfold = KFold(n_splits=n_splits, shuffle=True)

    # Iterate over kfolds
    rmse_folds = []
    rmse_folds_sum = [0, 0]
    for fold, (train_idx, val_idx) in enumerate(kfold.split(images)) if KFOLD else enumerate([(0,0)]):

        # Load flow field data
        if KFOLD:
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
        else:
            trainloader = DataLoader(
                dataset, 
                batch_size=config['train']['batch_size'], 
                pin_memory=True, 
                num_workers=config['train']['num_workers'],
            )
            testloader = DataLoader(
                dataset_test, 
                batch_size=config['train']['batch_size'], 
                pin_memory=True, 
                num_workers=config['train']['num_workers'],
            )

        # initialize_weights(gen)
        # initialize_weights(disc)

        # Training loop
        if KFOLD:
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
        else:
            print(
                f"  Training samples: {len(dataset)}\n"
                f"  Testing samples: {len(dataset_test)}\n"
                f"  Number of epochs: {NUM_EPOCHS}\n"
                f"  Mini batch size: {config['train']['batch_size']}\n"
                f"  Number of batches: {len(trainloader) if MULTIBATCH else 1}\n"
                f"  Learning rate: {config['train']['lr']}\n"
            )

        # TRAINING LOOP
        for epoch in range(NUM_EPOCHS):
            gen.train()
            # Iterate over mini batches
            rmse_tra_b = [0, 0]
            for image, inflow in trainloader if MULTIBATCH else [(0,0)]:

                if not MULTIBATCH:
                    image, inflow = next(iter(trainloader))

                image = image.float().to(device)
                inflow = inflow.float().to(device)

                torch.autograd.set_detect_anomaly(False)
                # Generate flow field prediction (synth image)
                synths = gen(inflow) # gen generates flow field image from inflow velocity

                ## Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                pred_real = disc(image, inflow) # Evaluate real data 
                pred_synth = disc(synths, inflow) # Evaluate synth data

                loss_real = criterion(pred_real, torch.ones_like(pred_real))
                loss_synth = criterion(pred_synth, torch.zeros_like(pred_synth))
                loss_d = (loss_synth + loss_real) / 2 # Combine losses

                opt_disc.zero_grad() # Reset gradients to zero
                loss_d.backward(retain_graph=True) # Backward pass
                opt_disc.step() # Update weights

                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                loss_g = criterion(disc(synths, inflow), torch.ones_like(disc(synths, inflow))) # Calculate loss

                opt_gen.zero_grad() # Reset gradients to zero
                loss_g.backward() # Backward pass
                opt_gen.step() # Update weights

                # Evaluate model on training data
                mse_tra = [0, 0]
                for (im_tra, synth_tra) in zip(image, synths):
                    if CHANNELS == 1:
                        mse_tra[0] += calc_mse(im_tra.detach(), synth_tra.detach())
                    elif CHANNELS == 2:
                        mse_tra[0] += calc_mse(im_tra.detach()[0], synth_tra.detach()[0])
                        mse_tra[1] += calc_mse(im_tra.detach()[1], synth_tra.detach()[1])

                mse_tra[0] /= len(image); rmse_tra_b[0] += mse_tra[0]
                mse_tra[1] /= len(image); rmse_tra_b[1] += mse_tra[1]
            
            if MULTIBATCH:
                rmse_tra_b[0] /= len(trainloader)
                rmse_tra_b[1] /= len(trainloader)

            # get rmse
            rmse_tra_b[0] = torch.sqrt(rmse_tra_b[0]).cpu().detach().numpy()
            if CHANNELS == 2:
                rmse_tra_b[1] = torch.sqrt(rmse_tra_b[1]).cpu().detach().numpy()

            # evaluation of the model on the testing data. This is for tuning hyperparameter to assess the regularization of the model
            gen.eval()
            rmse_test_b = [0, 0]
            # for each mini batch in testloader if multibatch
            for ims_test, inflow_test in testloader if MULTIBATCH else [(0,0)]:
                if not MULTIBATCH:
                    ims_test, inflow_test = next(iter(testloader))

                with torch.no_grad():
                    synths_test = gen(inflow_test.float().to(device))
                    # iterate over all images in this batch
                    mse_test = [0, 0]
                    for synth_test, im_test in zip(synths_test, ims_test):
                        # calculate mse
                        if CHANNELS == 1:
                            mse_test[0] += calc_mse(im_test.to(device), synth_test)
                        elif CHANNELS == 2:
                            mse_test[0] += calc_mse(im_test.to(device)[0], synth_test[0])
                            mse_test[1] += calc_mse(im_test.to(device)[1], synth_test[1])
                    mse_test[0] /= len(synths); rmse_test_b[0] += mse_test[0]
                    mse_test[1] /= len(synths); rmse_test_b[1] += mse_test[1]
                    
            if MULTIBATCH:
                rmse_test_b[0] /= len(testloader)
                rmse_test_b[1] /= len(testloader)

            # get rmse
            rmse_test_b[0] = torch.sqrt(rmse_test_b[0]).cpu().detach().numpy()
            if CHANNELS == 2:
                rmse_test_b[1] = torch.sqrt(rmse_test_b[1]).cpu().detach().numpy()

            if KFOLD:
                # Test model on validation data
                gen.eval()
                with torch.no_grad():
                    rmse_val = [0, 0]
                    # Iterate over minibatches of val data
                    for images, inflows in valloader:
                        # Generate images for this minibatch
                        images = images.float().to(device)
                        inflows = inflows.float().to(device)
                        ims_gen = gen(inflows)
                        # Iterate over all images in this batch
                        for image, im_gen in zip(images, ims_gen):
                            rmse_val[0] += calc_mse(image[0], im_gen[0])
                            if CHANNELS > 1:
                                rmse_val[1] += calc_mse(image[1], im_gen[1])
                        rmse_val[0] /= len(images)
                        rmse_val[1] /= len(images)
                    rmse_val[0] /= len(valloader)
                    rmse_val[1] /= len(valloader)
                    rmse_val[0] = torch.sqrt(rmse_val[0]).cpu().detach().numpy()
                    if CHANNELS > 1:
                        rmse_val[1] = torch.sqrt(rmse_val[1]).cpu().detach().numpy()
                gen.train()
            else:
                rmse_val=[]

            # Print progress every epoch
            if KFOLD:
                print(
                    f"Epoch [{epoch+1:03d}/{NUM_EPOCHS:03d} - "
                    f"Loss D_real: {loss_real:.3f}, Loss D_synth: {loss_synth:.3f}"
                    f", Loss G: {loss_g:.3f}, RMSE Tra.: ({rmse_tra_b[0]:.3f}, {rmse_tra_b[1]:.3f}),"
                    f" RMSE Val.: ({rmse_val[0]:.3f}, {rmse_val[1]:.3f})]"
                    )
            else:
                print(
                    f"Epoch [{epoch+1:03d}/{NUM_EPOCHS:03d} - "
                    f"Loss D_real: {loss_real:.3f}, Loss D_synth: {loss_synth:.3f}"
                    f", Loss G: {loss_g:.3f}, RMSE Tra.: ({rmse_tra_b[0]:.3f}, {rmse_tra_b[1]:.3f})"
                    f", RMSE Test: ({rmse_test_b[0]:.3f}, {rmse_test_b[1]:.3f})"
                    )
                        
            if epoch == 0:
                fig_im = plt.figure(dpi=300) # Initialize figure for flow field comparison
                grid = ImageGrid( # Create grid of images
                    fig_im,
                    111, 
                    nrows_ncols=(2,3) if CHANNELS == 2 else (2,3), 
                    axes_pad=0.15, 
                    share_all='True', 
                    cbar_location='right', 
                    cbar_mode='edge')
                # Initialize figure for loss and rmse evolution
                fig, axs = plt.subplots(2,1,dpi=300)
                # Initialize list containing metrics
                loss_disc_synth = []; loss_disc_real = []; loss_gen = []
                rmse_evol_ux_tra = []; rmse_evol_uy_tra = []
                rmse_evol_ux_val = []; rmse_evol_uy_val = []; rmse_evol_ux_test = []
                rmse_evol_uy_test = []

            # Plot metrics and flow field comparison
            plot_metrics(loss_disc_real, loss_disc_synth, loss_real, loss_synth, loss_gen, loss_g, epoch, fig, axs, NUM_EPOCHS, rmse_tra_b, rmse_test_b, rmse_val, rmse_evol_ux_tra, rmse_evol_uy_tra, rmse_evol_ux_test, rmse_evol_ux_test,  rmse_evol_ux_val, rmse_evol_uy_val)
            if KFOLD:
                # Plot validation images
                plot_flow_field_comparison(fig_im, grid, image, im_gen)
            else:
                # Plot training and testing images
                p = 0
                plot_flow_field_comparison(fig_im, grid, image[p].detach(), synths[p].detach(), ims_test[p].detach(), synths_test[p].detach().cpu())


        # Save models at the end of all epochs
        if config['models']['save']:
            # Save Generator
            torch.save({
                "state_dict": gen.state_dict(),
                "optimizer": opt_gen.state_dict()
            }, os.path.join('models', config['models']['name_gen']))
            # Save Discriminator
            torch.save({
                "state_dict": disc.state_dict(),
                "optimizer": opt_disc.state_dict()
            }, os.path.join('models', config['models']['name_disc']))
            print('Saving models...')

        if KFOLD:
            # Append RMSE to list of RMSE per fold
            rmse_folds.append(rmse_val)
            # Average RMSE of all folds
            print(f"\nAverage RMSE of fold {fold+1}/{n_splits}: ({rmse_folds[fold][0]:.2f}, {rmse_folds[fold][1]:.2f})")

            rmse_folds_sum[0] += rmse_folds[fold][0]
            rmse_folds_sum[1] += rmse_folds[fold][1]

    if KFOLD:
        print(
            f"\nAverage RMSE of all folds: "
            f"({rmse_folds_sum[0]/len(rmse_folds):.2f}, {rmse_folds_sum[1]/len(rmse_folds):.2f})")

    return

if __name__ == "__main__":
    tic = time.time()
    train()
    toc = time.time()

    print(f"Training duration: {((toc-tic)/60):.2f} m ")