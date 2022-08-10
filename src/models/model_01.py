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
    def __init__(self, channels, label_dim):
        super(Embedding, self).__init__()
        self.label_dim = label_dim
        self.channels = channels
        self.emb = nn.Linear(label_dim, label_dim**2)

    def forward(self, x):
        print(x.shape)
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
    def __init__(self, channels, height, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: (N,C,H,H), (472,2,44,44)
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), # (N,C,W,H), (472,2,44,44)
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
    N, C, H, W = 570, 2, 44, 44
    mu = torch.randn((N, C, H))
    x = torch.randn((N, C, H, W))
    emb = Embedding(C, H)
    print(f"Flow parameter shape: {mu.shape}")
    print(f"Embedding shape: {emb(mu).shape}")

    disc = Discriminator(C+2, 8)
    initialize_weights(disc)

    x_emb = torch.cat((emb(mu), x), 1)

    print(f"Cat shape: {x_emb.shape}")
    print(f"Discriminator shape: {disc(x_emb).shape}")

    gen = Generator(C, H, 8)
    initialize_weights(gen)
    # Generator receives image like inputs, we apply the embedding to the (N, C, H) flow parameter (mu) in order to pass to the Generator (instead of the usual random noise): (N, C, H, H).
    print(f"Generator shape: {gen(emb(mu)).shape}")


if __name__ == "__main__":
    test()