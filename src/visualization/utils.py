#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module with classes and functions related to visualization of results"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "08/22"

import os

import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from torch import sum, sqrt
import yaml

# Load config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

SIZE_LINEAR = config['data']['final_size'][0]*config['data']['final_size'][1]
CLIM_UX = config['data']['figures']['clim_ux']
CLIM_UY = config['data']['figures']['clim_uy']
CHANNELS = config['data']['channels']

def plot_metrics(loss_disc_real, loss_disc_fake, loss_real, loss_fake, loss_gen, loss_g, epoch, fig, axs, NUM_EPOCHS, rmse_tra, rmse_val, rmse_evol_ux_tra, rmse_evol_uy_tra, rmse_evol_ux_val, rmse_evol_uy_val):
    # Append current epoch loss to list of losses
    loss_disc_real.append(float(loss_real.detach().cpu()))
    loss_disc_fake.append(float(loss_fake.detach().cpu()))
    loss_gen.append(float(loss_g.detach().cpu()))

    rmse_evol_ux_tra.append(float(rmse_tra[0]))
    rmse_evol_ux_val.append(float(rmse_val[0]))
    if CHANNELS > 1:
        rmse_evol_uy_tra.append(float(rmse_tra[1]))
        rmse_evol_uy_val.append(float(rmse_val[1]))
    # Plot loss
    x = np.arange(0, epoch+1)

    axs[0].plot(x, loss_disc_real, label='Disc. loss (real)', color='k')
    axs[0].plot(x, loss_disc_fake, label='Disc. loss (fake)', color='b')
    axs[0].plot(x, loss_gen, label='Gen. loss', color='r')

    axs[1].plot(x, rmse_evol_ux_tra, label='RMSE Ux (Tra.)', color='g', ls='--')
    axs[1].plot(x, rmse_evol_ux_val, label='RMSE Ux (Val.)', color='g', ls='-')
    if CHANNELS > 1:
        axs[1].plot(x, rmse_evol_uy_tra, label='RMSE Uy (Tra.)', color='y', ls='--')
        axs[1].plot(x, rmse_evol_uy_val, label='RMSE Uy (Val.)', color='y', ls='-')
    
    axs[1].set(xlabel='epochs')
    axs[0].set(ylabel='loss')
    axs[1].set(ylabel='RMSE')
    
    axs[0].xaxis.set_ticklabels([])

    axs[0].set_ylim(0, 1.6)
    axs[1].set_ylim(0, 0.4)

    for i, ax in enumerate(axs):
        if epoch == 0:
            ax.set_xlim(1, NUM_EPOCHS-1)
            ax.legend(loc='upper right' if i == 1 else 'lower right', fontsize='x-small')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Save figure
        ax.grid(visible=True)
    fig.suptitle('Losses and RMSE through epochs')
    fig.savefig(os.path.join('figures', 'metrics.png'))

def calc_mse(image_real, image_fake):
    return sum((image_fake-image_real)**2)/SIZE_LINEAR

def plot_flow_field_comparison(fig_im, grid, image, im_gen):
    # Plot figure with images real and fake
    if CHANNELS == 1:
        grid_ims = [image[0], im_gen[0], im_gen[0]-image[0]]
    else:
        grid_ims = [image[0], im_gen[0], im_gen[0]-image[0],image[1], im_gen[1], im_gen[1]-image[1]]

    # Iterate over images in grid
    for c, (ax, im) in enumerate(zip(grid, grid_ims)):
        
        # Determine the image vmin and vmax
        if c < 3:
            im = im * (CLIM_UX[1]-CLIM_UX[0]) + CLIM_UX[0]
            vmin = CLIM_UX[0]; vmax = CLIM_UX[1]
        else:
            im = im * (CLIM_UY[1]-CLIM_UY[0]) + CLIM_UY[0]
            vmin = CLIM_UY[0]; vmax = CLIM_UY[1]

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
            ax.get_yaxis().set_visible(True)
            ax.set_ylabel("$Ux$")
            ax.set_title("U$_{real}$")
            ax.set_yticks([])
        elif c == 1:
            ax.set_title("U$_{fake}$")
        elif c == 2:
            ax.set_title("U$_{fake}$ - U$_{real}$")
            cb = grid.cbar_axes[0].colorbar(im)
            cb.set_label("[ms$^{-1}$]", rotation=270, labelpad=10)
        elif c == 3:
            ax.get_yaxis().set_visible(True)
            ax.set_ylabel("$Uy$")
            ax.set_yticks([])
        elif c == 5:
            cb = grid.cbar_axes[1].colorbar(im)
            cb.set_label("[ms$^{-1}$]", rotation=270, labelpad=18)

    # Add suptitle
    fig_im.suptitle(
        f"Flow field comparison (Validation data):\n"
        f"RMSE: ({sqrt(calc_mse(image[0], im_gen[0])):.3f}, {sqrt(calc_mse(image[1], im_gen[1])) if CHANNELS == 2 else 0.0:.3f})"
        )

    # Save figure
    fig_im.savefig(os.path.join('figures', 'image_comparison.png'), dpi=300)

    return fig_im