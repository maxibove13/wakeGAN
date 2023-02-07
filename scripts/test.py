#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to evaluate the wakeGAN generator on a pretrained model"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

from typing import Dict
import json
import os

from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
import pytorch_lightning as pl
import torch
import yaml

from src.data import dataset
from src.utils import callbacks
from src.wakegan import WakeGAN
from scripts import generate_wf

torch.set_float32_matmul_precision("medium")

with open("config.yaml") as file:
    config = yaml.safe_load(file)

tb_logger = TensorBoardLogger(save_dir="logs/other_logs")


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

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        max_epochs=config["train"]["num_epochs"],
        callbacks=[callbacks.PlottingCallback(neptune_run)],
        logger=loggers,
        deterministic=True,
        enable_checkpointing=False,
    )

    if config["ops"]["neptune_logger"]:
        neptune_run.log_hyperparams(params=config)

    dataset_train = dataset.WakeGANDataset(
        data_dir=os.path.join("data", "preprocessed", "tracked", "train"),
        config=config["data"],
        dataset_type="train",
        save_norm_params=True if config["models"]["save"] else False,
    )

    datamodule = dataset.WakeGANDataModule(config)
    model = WakeGAN(config, dataset_train.norm_params)

    trainer.validate(
        model=model,
        datamodule=datamodule,
        ckpt_path=_get_ckpt_path(),
    )
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=_get_ckpt_path(),
    )

    generate_wf.main()


def _get_ckpt_path():
    ckpt_path = os.path.join("logs", "lightning_logs")
    versions = os.listdir(ckpt_path)
    versions.sort()
    versions_number = [int(v.split("_")[-1]) for v in versions]
    versions_number.sort()
    versions = [f"version_{v}" for v in versions_number]
    ckpt_name = os.listdir(os.path.join(ckpt_path, versions[-1], "checkpoints"))[0]
    ckpt_path = os.path.join(ckpt_path, versions[-1], "checkpoints", ckpt_name)
    return ckpt_path


if __name__ == "__main__":
    main()
