#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module with classes and functions related to visualization of results"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "08/22"

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from torch import sum, sqrt
import torchvision.transforms as T
import yaml

# Load config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

SIZE_LINEAR = config['data']['final_size'][0]*config['data']['final_size'][1]
CLIM_UX = config['data']['figures']['clim_ux']
CLIM_UY = config['data']['figures']['clim_uy']
CHANNELS = config['data']['channels']
KFOLD = config['validation']['kfold']
MEAN = config['data']['norm']['mean']
STD = config['data']['norm']['std']

unnormalize = T.Normalize((-MEAN / STD), (1.0 / STD))

fig_err = plt.figure(dpi=300)
grid_err = ImageGrid( # Create grid of images
    fig_err,
    111, 
    nrows_ncols=(2,1), 
    axes_pad=0.15, 
    share_all='True', 
    cbar_location='right', 
    cbar_mode='edge')

def plot_metrics(loss_disc_real, loss_disc_fake, loss_real, loss_fake, loss_gen, loss_g, epoch, fig, axs, NUM_EPOCHS, rmse_tra, rmse_test, rmse_val, rmse_evol_ux_tra, rmse_evol_uy_tra, rmse_evol_ux_test, rmse_evol_uy_test, rmse_evol_ux_val, rmse_evol_uy_val):
    # Append current epoch loss to list of losses
    loss_disc_real.append(float(loss_real.detach().cpu()))
    loss_disc_fake.append(float(loss_fake.detach().cpu()))
    loss_gen.append(float(loss_g.detach().cpu()))

    rmse_evol_ux_tra.append(float(rmse_tra[0]))
    rmse_evol_ux_test.append(float(rmse_test[0]))
    if KFOLD:
        rmse_evol_ux_val.append(float(rmse_val[0]))
    if CHANNELS > 1:
        rmse_evol_uy_tra.append(float(rmse_tra[1]))
        if KFOLD:
            rmse_evol_uy_val.append(float(rmse_val[1]))
    # Plot loss
    x = np.arange(0, epoch+1)

    axs[0].plot(x, loss_disc_real, label='Discriminator loss (real)', color='k')
    axs[0].plot(x, loss_disc_fake, label='Discriminator loss (synth)', color='b')
    axs[0].plot(x, loss_gen, label='Generator loss', color='r')

    sec_ax1 = axs[1].secondary_yaxis('right', functions=(lambda x: x/CLIM_UX[1]*100, lambda x: x * CLIM_UX[1]*100))

    axs[1].plot(x, rmse_evol_ux_tra, label='RMSE Ux (Training)', color='g', ls='-')
    axs[1].plot(x, rmse_evol_ux_test, label='RMSE Ux (Testing)', color='g', ls='--')
    if KFOLD:
        axs[1].plot(x, rmse_evol_ux_val, label='RMSE Ux (Validation)', color='g', ls='-')
    if CHANNELS > 1:
        axs[1].plot(x, rmse_evol_uy_tra, label='RMSE Uy (Training)', color='y', ls='--')
        if KFOLD:
            axs[1].plot(x, rmse_evol_uy_val, label='RMSE Uy (Validation)', color='y', ls='-')
    
    axs[1].set(xlabel='epochs')
    axs[0].set(ylabel='loss')
    axs[1].set(ylabel="RMSE [ms$^{-1}$]")
    sec_ax1.set_ylabel('RMSE [% of range]', rotation=270, labelpad=14)
    
    axs[0].xaxis.set_ticklabels([])

    # axs[0].set_ylim(0, 1.6)
    axs[1].set_ylim(0, 2)

    for i, ax in enumerate(axs):
        if epoch == 0:
            ax.set_xlim(1, NUM_EPOCHS-1)
            ax.legend(loc='upper right' if i == 1 else 'lower right', fontsize='x-small')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Save figure
        ax.grid(visible=True)
    # fig.suptitle('Losses and RMSE through epochs')
    fig.savefig(os.path.join('figures', 'monitor', 'metrics.png'))

def calc_mse(real, synth):
    real = real * (CLIM_UX[1]-CLIM_UX[0]) + CLIM_UX[0]
    synth = synth * (CLIM_UX[1]-CLIM_UX[0]) + CLIM_UX[0]
    return sum((synth-real)**2)/SIZE_LINEAR

def plot_flow_field_comparison(fig_im, grid, image, im_gen, image_test, im_gen_test):
    # Plot figure with images real and fake
    if CHANNELS == 1:
        grid_ims = [image, im_gen, im_gen-image,
                    image_test, im_gen_test,im_gen_test - image_test
                    ]
    else:
        grid_ims = [image, im_gen, im_gen-image,image, im_gen, im_gen-image]
    grid_im_err = [im_gen-image, im_gen_test - image_test]

    # Iterate over images in grid
    for c, (ax, im) in enumerate(zip(grid, grid_ims)):
        
        # unnormalize
        if c <= 3:
            im = unnormalize(im)
        # remove extra dimension (channels, 1)
        im = im[0]


        # Determine the image vmin and vmax
        # if c < 3:
        im = im * (CLIM_UX[1]-CLIM_UX[0]) + CLIM_UX[0]
        vmin = CLIM_UX[0]; vmax = CLIM_UX[1]
        # else:
        #     im = im * (CLIM_UY[1]-CLIM_UY[0]) + CLIM_UY[0]
        #     vmin = CLIM_UY[0]; vmax = CLIM_UY[1]

        # Create image
        im = ax.imshow(
            im.detach().cpu(),
            cmap=cm.coolwarm, 
            interpolation='none',
            vmin=vmin, vmax=vmax)

        # Hide the axis
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Add titles, labels and colorbars
        if c == 0:
            ax.get_yaxis().set_visible(True)
            ax.set_ylabel("Training sample")
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
            ax.set_ylabel("Validation sample")
            ax.set_yticks([])
        elif c == 5:
            cb = grid.cbar_axes[1].colorbar(im)
            cb.set_label("[ms$^{-1}$]", rotation=270, labelpad=10)

    # Add suptitle
    # fig_im.suptitle(
    #     f"Flow field comparison (Validation data):\n"
    #     f"RMSE: ({sqrt(calc_mse(image[0], im_gen[0])):.3f}, {sqrt(calc_mse(image[1], im_gen[1])) if CHANNELS == 2 else 0.0:.3f})"
    #     )

    for c, (ax, im, cax) in enumerate(zip(grid_err, grid_im_err, grid_err.cbar_axes)):
        
        # unnormalize
        if c == 0:
            im = unnormalize(im)
        # remove extra dimension (channels, 1)
        im = im[0]


        # Determine the image vmin and vmax
        im = im * (CLIM_UX[1]-CLIM_UX[0]) + CLIM_UX[0]
        vmin = CLIM_UX[0]; vmax = CLIM_UX[1]

        # Create image
        im_plt = ax.imshow(
            im.detach().cpu(),
            cmap=cm.coolwarm, 
            interpolation='none',
            vmin=-1, vmax=1)

        # Hide the axis
        ax.get_xaxis().set_visible(False)

        # Add titles, labels and colorbars
        if c == 0:
            ax.set_ylabel("Training sample")
            ax.set_title("U$_{real}$")
            ax.set_yticks([])
            ax.set_title("U$_{fake}$ - U$_{real}$")
            # cb = grid.cbar_axes[0].colorbar(im)
        elif c == 1:
            ax.set_ylabel("Testing sample")
        # cb = grid.cbar_axes[c].colorbar(im)
        cbar = cax.colorbar(im_plt)
        cbar.set_label("[ms$^{-1}$]", rotation=270, labelpad=10)


    # Save figure
    fig_im.savefig(os.path.join('figures', 'monitor', 'image_comparison.png'), dpi=300)
    fig_err.savefig(os.path.join('figures', 'monitor', 'err_evol.png'), dpi=300)

    return fig_im