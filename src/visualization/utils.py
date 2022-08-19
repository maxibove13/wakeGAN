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
from torch import sum
import yaml

# Load config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

SIZE_LINEAR = config['data']['final_size'][0]*config['data']['final_size'][1]

def plot_metrics(loss_disc, loss_gen, loss_d, loss_g, epoch, fig, axs, num_epochs, rmse, rmse_evol_ux, rmse_evol_uy):
    # Append current epoch loss to list of losses
    loss_disc.append(float(loss_d.detach().cpu()))
    loss_gen.append(float(loss_g.detach().cpu()))
    rmse_evol_ux.append(float(rmse[0]))
    rmse_evol_uy.append(float(rmse[1]))
    # Plot loss
    x = np.arange(0, epoch+1)

    axs[0].plot(x, loss_disc, label='Discriminator loss', color='b')
    axs[0].plot(x, loss_gen, label='Generator loss', color='r')

    axs[1].plot(x, rmse_evol_ux, label='RMSE Ux', color='g')
    axs[1].plot(x, rmse_evol_uy, label='RMSE Uy', color='y')
    
    axs[1].set(xlabel='epochs')
    axs[0].set(ylabel='loss')
    axs[1].set(ylabel='RMSE')
    
    axs[0].xaxis.set_ticklabels([])

    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 0.5)

    for i, ax in enumerate(axs):
        if epoch == 0:
            ax.set_xlim(1, num_epochs-1)
            ax.legend(loc='upper right' if i == 1 else 'lower right')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Save figure
        ax.grid(visible=True)
    fig.suptitle('Losses and RMSE through epochs')
    fig.savefig(os.path.join('figures', 'metrics.png'))

def calc_mse(image_real, image_fake):
    return sum((image_fake-image_real)**2)/SIZE_LINEAR