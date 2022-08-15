#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module with classes and functions related to visualization of results"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "08/22"

import numpy as np
from matplotlib.ticker import MaxNLocator

def plot_losses(loss_disc, loss_gen, loss_d, loss_g, epoch, fig, ax, num_epochs):
    # Append current epoch loss to list of losses
    loss_disc.append(float(loss_d.detach().cpu()))
    loss_gen.append(float(loss_g.detach().cpu()))
    # Plot loss
    x = np.arange(0, epoch+1)
    ax.plot(x, loss_disc, label='Discriminator loss', color='b')
    ax.plot(x, loss_gen, label='Generator loss', color='r')
    ax.set_title('Evolution of losses through epochs')
    ax.set(xlabel='epochs')
    ax.set(ylabel='loss')
    ax.set_xlim(1, num_epochs-1)
    ax.set_ylim(0, 1.5)
    if epoch == 0:
        ax.legend(loc='upper right')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Save figure
    ax.grid(visible=True)
    fig.savefig('loss.png')

def plot_rmse(rmse, epoch, num_epochs, fig, ax, rmse_evol_ux, rmse_evol_uy):
    # Append current epoch loss to list of losses
    rmse_evol_ux.append(float(rmse[0]))
    rmse_evol_uy.append(float(rmse[1]))
    # Plot loss
    x = np.arange(0, epoch+1)
    ax.plot(x, rmse_evol_ux, label='RMSE Ux', color='g')
    ax.plot(x, rmse_evol_uy, label='RMSE Uy', color='y')
    ax.set_title('Evolution of RMSE through epochs')
    ax.set(xlabel='epochs')
    ax.set(ylabel='RMSE')
    ax.set_xlim(1, num_epochs-1)
    # ax.set_ylim(0, 1.5)
    if epoch == 0:
        ax.legend(loc='upper right')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Save figure
    ax.grid(visible=True)
    fig.savefig('rmse.png')