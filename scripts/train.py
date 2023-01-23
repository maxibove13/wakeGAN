#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to train the wakeGAN"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

import logging
import time
import os

import yaml

from scripts.evaluate import evaluate
from src.wakegan import WakeGAN

logging.basicConfig(
    format="%(message)s",
    filename=os.path.join("logs", "train.log"),
    level=logging.INFO,
    filemode="w",
)
logger = logging.getLogger("train")


def main():
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    with open("params.yaml") as file:
        hparams = yaml.safe_load(file)

    config["train"]["num_epochs"] = hparams["num_epochs"]
    config["train"]["lr"] = hparams["lr"]
    config["train"]["batch_size"] = hparams["batch_size"]
    config["train"]["f_adv_gen"] = hparams["f_adv_gen"]
    config["train"]["f_mse"] = hparams["f_mse"]

    wakegan = WakeGAN(config, logger)

    wakegan.set_device()
    wakegan.preprocess_dataset()
    wakegan.initialize_models()
    wakegan.define_loss_and_optimizer()
    if wakegan.load:
        wakegan.load_pretrained_models()
    wakegan.train()


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()

    logger.info(f"Training duration: {((toc-tic)/60):.2f} m ")
