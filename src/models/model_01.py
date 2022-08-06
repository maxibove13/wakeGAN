#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module that contains Discriminator and Generator classes"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "08/22"

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    # features_d are the channels that change as we go through the layers of the D
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1), # Not included in ZZ, following a vanilla DCGAN here
            self._block(features_d, features_d*2,kernel_size=4, stride=2, padding=1),
            self._block(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1),
            self._block(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(),
            nn.Sigmoid()
        )

    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # nn.BatchNorm2d(out_channels), # Not included in ZZ
            nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Modules):
    super(Generator, self).__init__()
    def __init__(self, z_dim, channels_img, features_g):
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,),
        )