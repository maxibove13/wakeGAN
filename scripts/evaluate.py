#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to evaluate the wakeGAN generator on a pretrained model"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

import json
import logging
import os
import time

import torch
import yaml

from src.wakegan import WakeGAN
from src.data.dataset import WakeGANDataset
from src.visualization.plots import FlowImagePlotter, ProfilesPlotter


def evaluate():

    logging.basicConfig(
        format="%(message)s",
        filename=os.path.join("logs", "evaluate.log"),
        level=logging.INFO,
        filemode="w",
    )
    logger = logging.getLogger("evaluate")

    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    tic = time.time()
    wakegan = WakeGAN(config, logger)
    wakegan.set_device()

    with open(os.path.join("data", "norm_params.json")) as f:
        norm_params = json.load(f)

    dataset = WakeGANDataset(
        data_dir=wakegan.data_dir["test"],
        config=wakegan.data_config,
        dataset_type="dev",
        norm_params=norm_params,
    )

    wakegan.initialize_models()
    wakegan.define_loss_and_optimizer()
    wakegan.load_pretrained_models()

    dataset.set_loader(
        batch_size=len(dataset), num_workers=wakegan.workers, shuffle=False
    )

    images_raw, synths_raw, rmse = wakegan.evaluate_generator(dataset)

    images = torch.zeros((len(dataset), 1, wakegan.size[0], wakegan.size[1]))
    synths = torch.zeros((len(dataset), 1, wakegan.size[0], wakegan.size[1]))
    for c, (img, synth) in enumerate(zip(images_raw, synths_raw)):
        images[c] = wakegan.transform_back(img, dataset)
        synths[c] = wakegan.transform_back(synth, dataset)

    images = images.squeeze()
    synths = synths.squeeze()

    images_to_plot = [
        images[0],
        synths[0],
        images[1],
        synths[1],
        images[2],
        synths[2],
        images[3],
        synths[3],
    ]

    flow_image_plotter = FlowImagePlotter(
        channels=wakegan.channels, clim=dataset.clim, monitor=False, rmse=rmse
    )
    flow_image_plotter.plot(images_to_plot)

    profiles_plotter = ProfilesPlotter(
        config["data"]["wt_diam"],
        config["data"]["lim_around_wt"],
        config["data"]["size"],
    )
    profiles_plotter.plot(images_to_plot)

    toc = time.time()
    logger.info(f"Evaluation duration: {((toc-tic)/60):.2f} m ")


if __name__ == "__main__":
    evaluate()
