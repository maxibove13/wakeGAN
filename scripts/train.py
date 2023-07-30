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

# import neptune.new as neptune
import pytorch_lightning as pl
import torch
import yaml

from src.data import dataset
from src.utils import callbacks
from src.wakegan import WakeGAN

torch.set_float32_matmul_precision("medium")

# custom loggers
if "logs" not in os.listdir("."):
    os.mkdir("logs")

logging.basicConfig(
    format="%(message)s",
    filename=os.path.join("logs", "train.log"),
    level=logging.INFO,
    filemode="w",
)
logger = logging.getLogger("train")
tb_logger = TensorBoardLogger(save_dir="logs/")

# load hyperparameters config
with open("config.yaml") as file:
    config = yaml.safe_load(file)


def main():
    # initialize neptune client
    neptune_run = None
    if config["ops"]["neptune_logger"]:
        neptune_run = NeptuneLogger(
            project="idatha/wakegan",
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNWQ5YjJjZi05OTE1LTRhNWEtODdlZC00MWRlMzMzNGMwMzYifQ==",
            log_model_checkpoints=False,
        )

    loggers = (
        [tb_logger, neptune_run] if config["ops"]["neptune_logger"] else [tb_logger]
    )

    root_dir = os.path.join("data", "generated")
    folders = [
        os.path.join("testing", "real"),
        os.path.join("testing", "synth"),
        os.path.join("validation", "real"),
        os.path.join("validation", "synth"),
    ]
    if not os.path.exists(root_dir):
        for folder in folders:
            os.makedirs(os.path.join(root_dir, folder))

    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        save_top_k=1,
        monitor="rmse_val_epoch",
        mode="min",
        filename="wakegan-{epoch}-{rmse_val_epoch:.2f}",
    )

    # initialize dataset
    dataset_train = dataset.WakeGANDataset(
        data_dir=os.path.join("data", "preprocessed", "tracked", "train"),
        config=config["data"],
        dataset_type="train",
        save_norm_params=True if config["models"]["save"] else False,
    )
    datamodule = dataset.WakeGANDataModule(config)

    # initialize model
    model = WakeGAN(config, dataset_train.norm_params)

    # initialize trainer
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

    # log hyperparameters in neptune
    if config["ops"]["neptune_logger"]:
        neptune_run.log_hyperparams(params=config)

    # fit model
    trainer.fit(model, datamodule)

    # save model version (checkpoint) in neptune
    create_new_model_version(trainer, neptune_run)

    # validate and test model
    trainer.validate(model, datamodule.val_dataloader(), ckpt_path="best")
    trainer.test(model, datamodule.test_dataloader(), ckpt_path="best")

    # stop neptune run
    if neptune_run:
        neptune_run.run.stop()


def create_new_model_version(trainer, neptune_logger):
    if config["ops"]["neptune_logger"] and config["models"]["save"]:
        logger.info("Saving model in neptune")

        model_version = neptune.init_model_version(
            model="WAK-MOD",
            project="idatha/wakegan",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNWQ5YjJjZi05OTE1LTRhNWEtODdlZC00MWRlMzMzNGMwMzYifQ==",  # your credentials
        )
        path_to_model = trainer.checkpoint_callback.best_model_path
        model_version["model/ckpt"].upload(path_to_model)
        model_version["model/dataset/training"].track_files(
            os.path.join("data", "preprocessed", "tracked", "train", "ux")
        )
        model_version["model/dataset/validation"].track_files(
            os.path.join("data", "preprocessed", "tracked", "val", "ux")
        )
        model_version["model/dataset/testing"].track_files(
            os.path.join("data", "preprocessed", "tracked", "test", "ux")
        )
        model_version["model/run"] = neptune_logger.run["sys/id"].fetch()
        model_version.change_stage("staging")


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    logger.info(f"Training duration: {((toc-tic)/60):.2f} m ")
