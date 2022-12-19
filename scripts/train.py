#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to train the wakeGAN"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

import time

import yaml

from src.wake_gan import WakeGAN
from src.utils.logger import logger


def main():
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    wakegan = WakeGAN(config)

    wakegan.set_device()
    wakegan.preprocess_dataset()
    wakegan.initialize_models()


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()

    logger.info(f"Training duration: {((toc-tic)/60):.2f} m ")
