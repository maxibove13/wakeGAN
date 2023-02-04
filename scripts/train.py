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

from pytorch_lightning.callbacks import ModelCheckpoint, BatchSizeFinder
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
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

with open("config.yaml") as file:
    config = yaml.safe_load(file)

tb_logger = TensorBoardLogger(save_dir="logs/")


torch.set_float32_matmul_precision("medium")


def main():
    if config["ops"]["neptune_logger"]:
        neptune_run = NeptuneLogger(
            project="idatha/wakegan",
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNWQ5YjJjZi05OTE1LTRhNWEtODdlZC00MWRlMzMzNGMwMzYifQ==",
            log_model_checkpoints=False,
        )
    else:
        neptune_run = None

    loggers = (
        [tb_logger, neptune_run] if config["ops"]["neptune_logger"] else [tb_logger]
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        save_top_k=1,
        monitor="rmse_val_epoch",
        mode="min",
        filename="wakegan-{epoch}-{rmse_val_epoch:.2f}",
    )

    dataset_train = dataset.WakeGANDataset(
        data_dir=os.path.join("data", "preprocessed", "tracked", "train"),
        config=config["data"],
        dataset_type="train",
        save_norm_params=True if config["models"]["save"] else False,
    )
    datamodule = dataset.WakeGANDataModule(config)

    model = WakeGAN(config, dataset_train.norm_params)

    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        max_epochs=config["train"]["num_epochs"],
        logger=loggers,
        deterministic=True,
        callbacks=[
            callbacks.LoggingCallback(logger),
            callbacks.PlottingCallback(enable_logger=config["ops"]["neptune_logger"]),
            checkpoint_callback,
        ],
    )

    if config["ops"]["neptune_logger"]:
        neptune_run.log_hyperparams(params=config)

    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule.val_dataloader(), ckpt_path="best")
    trainer.test(model, datamodule.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    logger.info(f"Training duration: {((toc-tic)/60):.2f} m ")
