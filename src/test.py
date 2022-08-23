#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to test predictions on wind farm flow fields comparing them with LES simulations"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "08/22"

# Built-in modules
import os
import time

# Third party modules
import torch
from torch.utils.data import DataLoader
from torch import sum, sqrt
from matplotlib import cm, pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import yaml

# Local modules
from data.utils import load_prepare_dataset, ProcessedDataset
from models.model_01 import Generator
from visualization.utils import calc_mse, plot_flow_field_comparison

root_dir = os.path.join('data', 'preprocessed', 'test')

# Load config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

CHANNELS = config['data']['channels']
CLIM_UX = config['data']['figures']['clim_ux']
CLIM_UY = config['data']['figures']['clim_uy']

def test():

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
        f"Loading testing data with {images.shape[0]} samples\n"
        f"Flow parameters shape (N,C,H): {inflow.shape}\n" # (N,2,64)
        f"Flow field data shape (N,C,H,W): {images.shape}\n" # (N,2,64,64)
        )

    # Load image dataset in the pytorch manner
    dataset = ProcessedDataset(images, inflow)

    # Define Generator
    gen = Generator(CHANNELS, config['data']['final_size'][0], config['models']['f_g']).to(device)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))

    # Load Generator
    print('Loading pretrained Generator...\n')
    checkpoint_gen = torch.load(
        os.path.join('models', config['models']['name_gen']),
        map_location = device
    )
    gen.load_state_dict(checkpoint_gen["state_dict"])
    opt_gen.load_state_dict(checkpoint_gen["optimizer"])
    for param_group in opt_gen.param_groups:
        param_group["lr"] = config['train']['lr']

    # Load testing data in batches
    testloader = DataLoader(
    dataset, 
    batch_size=config['train']['batch_size'], 
    pin_memory=True, 
    num_workers=config['train']['num_workers'])

    rmse = [0, 0]
    gen.eval()
    with torch.no_grad():
        for (images, labels) in testloader:

            images = images.float().to(device)
            labels = labels.float().to(device)

            # Generate fake image
            fakes = gen(labels)
            # Iterate over all images in this minibatch
            for (image, fake) in zip(images, fakes):
                rmse[0] += calc_mse(image[0], fake[0])
            rmse[0] /= len(images)
        rmse[0] /= len(testloader)
        rmse[0] = torch.sqrt(rmse[0]).cpu().detach().numpy()


    fig_im = plt.figure(dpi=300) # fig for flow field comparison
    fig_err = plt.figure(dpi=300) # fig for error comparison
    grid = ImageGrid(
        fig_im, 
        111, 
        nrows_ncols=(2,4), 
        axes_pad=(0.3, 0.2), 
        share_all='False',
        label_mode='all', 
        cbar_location='right', 
        cbar_mode='edge',
        cbar_pad=0.15)
    grid_err = ImageGrid(
        fig_err, 
        111, 
        nrows_ncols=(1,4) if CHANNELS == 1 else (2,4),
        axes_pad=0.15, 
        share_all='True', 
        cbar_location='right', 
        cbar_mode='edge')

        # Plot figure with images real and fake
    if CHANNELS == 1:
        grid_ims = [
            images[0][0], fakes[0][0], images[1][0], fakes[1][0],
            images[2][0], fakes[2][0], images[3][0], fakes[3][0]]
        grid_err_ims = [
            fakes[0][0]-images[0][0],
            fakes[1][0]-images[1][0],
            fakes[2][0]-images[2][0],
            fakes[3][0]-images[3][0]]
    else:
        grid_ims = [
            images[0][0], fakes[0][0], fakes[0][0]-images[0][0],
            images[1][0], fakes[1][0], fakes[1][0]-images[1][0],
            images[2][0], fakes[2][0], fakes[2][0]-images[2][0],
            images[3][0], fakes[3][0], fakes[3][0]-images[3][0]]
        grid_err_ims = [
            fakes[0][0]-images[0][0],
            fakes[1][0]-images[1][0],
            fakes[2][0]-images[2][0],
            fakes[3][0]-images[3][0]]

    # Iterate over images in grid
    for c, (ax, im) in enumerate(zip(grid, grid_ims)):
        
        # Determine the image vmin and vmax
        im = im * (CLIM_UX[1]-CLIM_UX[0]) + CLIM_UX[0]
        vmin = CLIM_UX[0]; vmax = CLIM_UX[1]

        # Create image
        im = ax.imshow(
            im.cpu(),
            cmap=cm.coolwarm, 
            interpolation='none',
            vmin=vmin, vmax=vmax)

        # Hide the axis
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Add titles, labels and colorbars
        if c == 0:
            ax.set_title("U$^{real}$")
            ax.get_yaxis().set_visible(True)
            ax.set_ylabel(f"$Ux$ (Case #{c//2 + 1})")
            ax.set_yticks([])
        elif c == 1:
            ax.set_title("U$^{fake}$")
        elif c == 2:
            ax.set_title("U$^{real}$")
            ax.get_yaxis().set_visible(True)
            ax.set_ylabel(f"$Ux$ (Case #{c//2 + 1})")
            ax.set_yticks([])
        elif c in [4, 6]:
            ax.get_yaxis().set_visible(True)
            ax.set_ylabel(f"$Ux$ (Case #{c//2 + 1})")
            ax.set_yticks([])
        elif c == 3:
            ax.set_title("U$^{fake}$")
            cb = grid.cbar_axes[0].colorbar(im)
            cb.set_label("[ms$^{-1}$]", rotation=270, labelpad=10)
        elif c == 7:
            cb = grid.cbar_axes[1].colorbar(im)
            cb.set_label("[ms$^{-1}$]", rotation=270, labelpad=10)

    for c, (ax, im) in enumerate(zip(grid_err, grid_err_ims)):

        # Determine the image vmin and vmax
        im = im * (CLIM_UX[1]-CLIM_UX[0]) + CLIM_UX[0]

        # Create image
        im = ax.imshow(
            im.cpu(),
            cmap=cm.coolwarm, 
            interpolation='none',
            vmin=-1, vmax=1
            )

        cb = grid_err.cbar_axes[c].colorbar(im)
        cb.set_label("[ms$^{-1}$]", rotation=270, labelpad=10)
        ax.set_ylabel("$U_x^{fake} - U_x^{real}$")
        ax.set_yticks([])
        ax.set_title(f"Case #{c+1}")
        ax.get_xaxis().set_visible(False)

    # Add suptitle
    fig_im.suptitle(
        f"Flow field comparison\n"
        f"Average RMSE for testing dataset: ({sqrt(calc_mse(image[0], fake[0])):.3f}, {sqrt(calc_mse(image[1], fake[1])) if CHANNELS == 2 else 0.0:.3f})",
        y=0.9)

    # Add suptitle
    fig_err.suptitle(
        f"Flow field error comparison\n"
        f"Average RMSE for testing dataset: ({sqrt(calc_mse(image[0], fake[0])):.3f}, {sqrt(calc_mse(image[1], fake[1])) if CHANNELS == 2 else 0.0:.3f})",
        y=0.75)

    # Save figures
    fig_im.savefig(os.path.join('figures', 'image_comparison_test.png'), dpi=300, bbox_inches='tight')
    fig_err.savefig(os.path.join('figures', 'image_comparison_err_test.png'), dpi=300, bbox_inches='tight')

    print(f"RMSE for testing data: {rmse[0]:.3f}\n")

    return

if __name__ == "__main__":
    tic = time.time()
    test()
    toc = time.time()

    print(f"Testing evaluation duration: {((toc-tic)/60):.2f} m ")