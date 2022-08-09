#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module that contains Discriminator and Generator classes"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "08/22"

import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, channels_img, label_dim):
        super(Embedding, self).__init__()
        self.channels = channels_img
        self.label_dim = label_dim

        self.emb = nn.Linear(label_dim, label_dim*label_dim)

    def forward(self, x):
        x = self.emb(x)
        x = torch.reshape(x, (x.shape[0], self.channels, self.label_dim, self.label_dim))
        return x

class Discriminator(nn.Module):
    # features_d are the channels that change as we go through the layers of the D
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1), # Not included in ZZ, following a vanilla DCGAN here
            self._block(features_d, features_d*2,kernel_size=4, stride=2, padding=1),
            self._block(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1),
            self._block(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1),
            nn.Flatten(), # ZZ paper
            nn.Linear(features_d*32,1), # ZZ paper
            # nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0), # Vanilla DCGAN (not in ZZ paper)
            nn.Sigmoid()
        )
        # self.embed = nn.Embedding(num_classes, img_size*img_size)

    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # nn.BatchNorm2d(out_channels), # Not included in ZZ
            nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        x = self.disc(x)
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input N x z_dim x 1 x 1
            self._block(z_dim, features_g*16, 4, 1, 0), # N x f_g*16 x 4 x 4
            self._block(features_g*16, features_g*8, 4, 2, 1), # 8x8
            self._block(features_g*8, features_g*4, 4, 2, 1), # 16x16
            self._block(features_g*4, features_g*2, 4, 2, 1), # 16x16
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), # [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # nn.BatchNorm2d(out_channels), # Not included ni ZZ
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 570, 2, 44, 44
    z_dim = 100
    mu = torch.randn((N, in_channels, H))
    x = torch.randn((N, in_channels, H, W))
    emb = Embedding(in_channels, H)
    print(f"Flow parameter shape: {mu.shape}")
    print(f"Embedding shape: {emb(mu).shape}")

    disc = Discriminator(in_channels+2, 8)
    initialize_weights(disc)

    x_emb = torch.cat((emb(mu), x), 1)

    print(f"Cat shape: {x_emb.shape}")
    print(f"Discriminator shape: {disc(x_emb).shape}")

    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    print(f"Generator shape: {gen(z).shape}")


if __name__ == "__main__":
    test()