#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to train the wakeGAN"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

import logging
import os
import time

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
import yaml

from src.data import dataset
from src.utils import callbacks
from src.wakegan import WakeGAN

logging.basicConfig(
    format="%(message)s",
    filename=os.path.join("logs", "train.log"),
    level=logging.INFO,
    filemode="w",
)
logger = logging.getLogger("train")

torch.set_float32_matmul_precision("medium")


def main():
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    dataset, datamodule, trainer = init(config)

    model = WakeGAN(config, dataset.norm_params)

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule.val_dataloader(), ckpt_path="best")


def init(config):

    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        save_top_k=1,
        monitor="rmse_dev_epoch",
        mode="min",
        filename="wakegan-{epoch}-{rmse_dev_epoch:.2f}",
    )

    dataset_train = dataset.WakeGANDataset(
        data_dir=os.path.join("data", "preprocessed", "tracked", "train"),
        config=config["data"],
        dataset_type="train",
        save_norm_params=True if config["models"]["save"] else False,
    )

    datamodule = dataset.WakeGANDataModule(config)

    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        max_epochs=config["train"]["num_epochs"],
        callbacks=[
            callbacks.PlottingCallback(),
            callbacks.LoggingCallback(logger),
            checkpoint_callback,
        ],
    )

    return dataset_train, datamodule, trainer


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    logger.info(f"Training duration: {((toc-tic)/60):.2f} m ")
