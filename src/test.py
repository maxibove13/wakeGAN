#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to test predictions on wind farm flow fields comparing them with LES simulations"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "08/22"

# Built-in modules
import os
import time

# Third party modules
import torch
from torch.utils.data import DataLoader
import yaml

# Local modules
from data.utils import load_prepare_dataset, ProcessedDataset
from models.model_01 import Generator
from visualization.utils import calc_mse

root_dir = os.path.join('data', 'preprocessed', 'test')

# Load config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

CHANNELS = config['data']['channels']

def test():

    # Set device
    device = torch.device(config['train']['device']) if torch.cuda.is_available() else 'cpu'
    print(
        f"\n"
        f"Using device: {torch.cuda.get_device_name()}"
        f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB")

    # Load and process images, extract inflow velocity 
    images, inflow = load_prepare_dataset(root_dir)
    n_test_data = len(os.listdir(os.path.join('data', 'preprocessed', 'test', 'ux')))
    print(
        f"\n"
        f"Loading testingdata with {images.shape[0]} samples\n"
        f"Flow parameters shape (N,C,H): {inflow.shape}\n" # (N,2,64)
        f"Flow field data shape (N,C,H,W): {images.shape}\n" # (N,2,64,64)
        )

    # Load image dataset in the pytorch manner
    dataset = ProcessedDataset(images, inflow)

    # Define Generator
    gen = Generator(CHANNELS, config['data']['final_size'][0], config['models']['f_g']).to(device)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=config['train']['lr'], betas=(0.5, 0.999))

    # Load Generator
    print('Loading pretrained Generator')
    checkpoint_gen = torch.load(
        os.path.join('models', config['models']['name_gen']),
        map_location = device
    )
    gen.load_state_dict(checkpoint_gen["state_dict"])
    opt_gen.load_state_dict(checkpoint_gen["optimizer"])
    for param_group in opt_gen.param_groups:
        param_group["lr"] = config['train']['lr']

    # Load testing data in batches
    testloader = DataLoader(
    dataset, 
    batch_size=config['train']['batch_size'], 
    pin_memory=True, 
    num_workers=config['train']['num_workers'])

    rmse = [0, 0]
    gen.eval()
    with torch.no_grad():
        for (images, labels) in testloader:

            images = images.float().to(device)
            labels = labels.float().to(device)

            # Generate fake image
            fakes = gen(labels)
            # Iterate over all images in this minibatch
            for (image, fake) in zip(images, fakes):
                rmse[0] += calc_mse(image[0], fake[0])
            rmse[0] /= len(images)
        rmse[0] /= len(testloader)
        rmse[0] = torch.sqrt(rmse[0]).cpu().detach().numpy()

    print(rmse[0])

    return

if __name__ == "__main__":
    tic = time.time()
    test()
    toc = time.time()

    print(f"Training duration: {((toc-tic)/60):.2f} m ")