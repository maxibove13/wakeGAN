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

from scripts import evaluate
from src.wakegan import LitWakeGAN
from src.data import dataset
import pytorch_lightning as pl
from torch import set_float32_matmul_precision

from argparse import ArgumentParser


logging.basicConfig(
    format="%(message)s",
    filename=os.path.join("logs", "train.log"),
    level=logging.INFO,
    filemode="w",
)
logger = logging.getLogger("train")

set_float32_matmul_precision("medium")


def main():
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    # wakegan = WakeGAN(config, logger)

    # wakegan.set_device()
    # wakegan.preprocess_dataset()
    # wakegan.initialize_models()
    # wakegan.define_loss_and_optimizer()
    # if wakegan.load:
    #     wakegan.load_pretrained_models()
    # wakegan.train()

    # evaluate.evaluate()
    dataset_train = dataset.WakeGANDataset(
        data_dir=os.path.join("data", "preprocessed", "tracked", "train"),
        config=config["data"],
        save_norm_params=True if config["models"]["save"] else False,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10,
        max_epochs=config["train"]["num_epochs"],
    )

    datamodule = dataset.WakeGANDataModule(config)

    model = LitWakeGAN(
        config,
        dataset_train.norm_params,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    # args = parser.parse_args()

    tic = time.time()
    main()
    toc = time.time()

    logger.info(f"Training duration: {((toc-tic)/60):.2f} m ")
