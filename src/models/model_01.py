#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module that contains Discriminator and Generator classes"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "08/22"

import torch
import torch.nn as nn
import yaml

# Load config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

CHANNELS = config['data']['channels']

class Embedding(nn.Module):
    def __init__(self, channels, label_dim):
        super(Embedding, self).__init__()
        self.label_dim = label_dim
        self.channels = channels
        self.emb = nn.Linear(label_dim, label_dim**2)

    def forward(self, x):
        x = self.emb(x)
        x = torch.reshape(x, (x.shape[0], self.channels, self.label_dim, self.label_dim))
        return x

class Discriminator(nn.Module):
    def __init__(self, channels, features_d, height):
        super(Discriminator, self).__init__()
        self.height = height
        self.channels = channels
        self.linear = nn.Linear(height, height**2)
        self.disc = nn.Sequential(
            # Input: (N,4,64,64)
            nn.Conv2d(channels*2, features_d, 4, 2, 1), # (N,8,32,32)
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d, features_d*2, 4, 2, 1), # (N,16,16,16)
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d*2, features_d*4, 4, 2, 1), # (N,32,8,8)
            nn.LeakyReLU(0.2),
            nn.Flatten(), # (N,2048)
            nn.Linear(features_d*256, 1), # (N, 1)
            nn.Sigmoid()
        )

    # x is the image, mu is the inflow velocity (condition)
    def forward(self, x, mu):
        mu = self.linear(mu) 
        mu = torch.reshape(mu, (mu.shape[0], self.channels, self.height, self.height))
        x_emb = torch.cat((x, mu), 1)
        x_emb = self.disc(x_emb)
        return x_emb

class Generator(nn.Module):
    def __init__(self, channels, height, features_g):
        super(Generator, self).__init__()
        self.height = height
        self.channels = channels
        self.linear = nn.Linear(height, height*64//CHANNELS)
        self.gen = nn.Sequential(
            # Input: (N, H, 8, 8)
            nn.ConvTranspose2d(height, features_g*16, 4, 2, 1),  # (N,f_g*16,16,16)
            nn.LeakyReLU(),
            nn.ConvTranspose2d(features_g*16, features_g*8, 4, 2, 1), # (N,f_g*8,32,32)
            nn.LeakyReLU(),
            nn.ConvTranspose2d(features_g*8, channels, 4, 2, 1), # (N,C,W,H), (472,2,64,64)
            nn.Tanh(), # [-1, 1]
        )

    def forward(self, x): # (N,C,H)
        x = self.linear(x) # (N,C,H*64/C)
        x = torch.reshape(x, (x.shape[0], self.height, 8, 8)) # (N,H,8,8)
        x = self.gen(x) # (N,C,W,H)
        return x

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, C, H, W = 472, CHANNELS, config['data']['final_size'][1], config['data']['final_size'][0]
    mu = torch.randn((N, C, H))
    x = torch.randn((N, C, H, W))
    # emb = Embedding(C, H)
    print(f"Flow parameter shape: {mu.shape}")
    # print(f"Embedding shape: {emb(mu).shape}")

    disc = Discriminator(C, config['models']['f_d'], H)

    # x_emb = torch.cat((x, emb(mu)), 1)

    # print(f"Cat shape: {x_emb.shape}")
    print(f"Discriminator shape: {disc(x, mu).shape}")

    gen = Generator(C, H, 16)
    initialize_weights(gen)
    # Generator receives (N, C, H) flow parameter (mu)
    print(f"Generator shape: {gen(mu).shape}")

    print(sum(p.numel() for p in gen.parameters()))
    print(sum(p.numel() for p in disc.parameters()))
    # print(sum(p.numel() for p in emb.parameters()))


if __name__ == "__main__":
    test()