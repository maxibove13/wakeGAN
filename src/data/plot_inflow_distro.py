#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to plot the inflow velocity distribution of the preprocessed data"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "03/22"

import os

import numpy as np
from matplotlib import pyplot as plt
import yaml

from utils import load_prepare_dataset

# Load config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

CLIM_UX = config['data']['figures']['clim_ux']
CLIM_UY = config['data']['figures']['clim_uy']

class Inflow():
    def __init__(self, root_dir):
        _, self.tensor = load_prepare_dataset(root_dir)
        self.size = self.tensor.shape[-1]

    def unnormalize(self):
        self.tensor[:, 0, :] = self.tensor[:, 0, :] * (CLIM_UX[1]-CLIM_UX[0]) + CLIM_UX[0]


    def plot_wind_profiles(self, step=60):

        fig, ax = plt.subplots(1,1)

        for i in range(0, len(self.tensor), step):
            ax.plot(self.tensor[i, 0, :], np.arange(0, self.size), ls='--')

        ax.grid()
        ax.set_xlabel('$U_x$ [ms$^{-1}$]')
        ax.set_ylabel('Pixel position')
        ax.set_ylim(0, self.size)

        fig.savefig(
            os.path.join('figures','inflow_wind_profiles.png'),
            dpi=300,
            bbox_inches='tight'
        )

root_dir = os.path.join('data', 'preprocessed', 'train')

inflow = Inflow(root_dir)


# plot inflow wind profile
inflow.unnormalize()
inflow.plot_wind_profiles(step=60)