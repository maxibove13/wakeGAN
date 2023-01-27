#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module that contains Discriminator and Generator classes of a DCGAN"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

import torch
import torchsummary
import yaml


class Discriminator(torch.nn.Module):
    def __init__(self, channels, features_d, height):
        super(Discriminator, self).__init__()
        self.height = height
        self.loss = []
        self.channels = channels
        self.linear = torch.nn.Linear(height, height**2)
        self.disc = torch.nn.Sequential(
            # Input: (N,4,64,64)
            torch.nn.Conv2d(channels * 2, features_d, 4, 2, 1),  # (N,8,32,32)
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(features_d, features_d * 2, 4, 2, 1),  # (N,16,16,16)
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1),  # (N,32,8,8)
            torch.nn.LeakyReLU(0.2),
            torch.nn.Flatten(),  # (N,2048)
            torch.nn.Linear(features_d * 256, 1),  # (N, 1)
            torch.nn.Sigmoid(),
        )
        self.initialize_weights()

    # x is the image, mu is the inflow velocity (condition)
    def forward(self, x, mu):
        mu = self.linear(mu)
        mu = torch.reshape(mu, (mu.shape[0], self.channels, self.height, self.height))
        x_emb = torch.cat((x, mu), 1)
        x_emb = self.disc(x_emb)
        return x_emb

    def initialize_weights(self):
        gain_conv2d = torch.nn.init.calculate_gain("conv2d")
        gain_linear = torch.nn.init.calculate_gain("linear")
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=gain_conv2d)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=gain_linear)
                torch.nn.init.constant_(m.bias, 0)


class Generator(torch.nn.Module):
    def __init__(self, channels, height, features_g):
        super(Generator, self).__init__()
        self.height = height
        self.loss_adv = []
        self.loss_mse = []
        self.channels = channels
        self.linear = torch.nn.Linear(height, height * 64 // channels)
        self.gen = torch.nn.Sequential(
            # Input: (N, H, 8, 8)
            torch.nn.ConvTranspose2d(
                height, features_g * 16, 4, 2, 1
            ),  # (N,f_g*16,16,16)
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                features_g * 16, features_g * 8, 4, 2, 1
            ),  # (N,f_g*8,32,32)
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                features_g * 8, channels, 4, 2, 1
            ),  # (N,C,W,H), (472,2,64,64)
            torch.nn.Tanh(),  # [-1, 1]
        )
        self.initialize_weights()

    def forward(self, x):  # (N,C,H)
        x = self.linear(x)  # (N,C,H*64/C)
        x = torch.reshape(x, (x.shape[0], self.height, 8, 8))  # (N,H,8,8)
        x = self.gen(x)  # (N,C,W,H)
        return x

    def initialize_weights(self):
        gain_leaky = torch.nn.init.calculate_gain("leaky_relu", 1e-2)
        gain_linear = torch.nn.init.calculate_gain("linear")
        gain_convtrans = torch.nn.init.calculate_gain("conv_transpose2d")
        for m in self.modules():
            if isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight, gain=gain_convtrans)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=gain_linear)
                torch.nn.init.constant_(m.bias, 0)


def main():
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    N, C, H, W = (
        472,
        int(config["data"]["channels"]),
        config["data"]["size"][1],
        config["data"]["size"][0],
    )
    mu = torch.randn((N, C, H))
    x = torch.randn((N, C, H, W))
    print(f"inflow shape / velocity profile: {mu.shape}")

    disc = Discriminator(C, config["models"]["f_d"], H)
    print(f"Discriminator shape / binary classifier output: {disc(x, mu).shape}")

    # Generator receives (N, C, H) flow parameter (mu)
    gen = Generator(C, H, config["models"]["f_d"])
    print(f"Generator shape / velocity planes images: {gen(mu).shape}")

    # print("parameters of gen: ", sum(p.numel() for p in gen.parameters()))
    # print("parameters of disc: ", sum(p.numel() for p in disc.parameters()))
    torchsummary.summary(gen.cuda(), (C, H))
    torchsummary.summary(disc.cuda(), [(1, 64, 64), (1, 64)], batch_size=32)


if __name__ == "__main__":
    main()
