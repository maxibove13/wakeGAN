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

import pytorch_lightning as pl
import torch
import yaml

from src.data import dataset
from src.utils import callbacks
from src.wakegan import WakeGAN

torch.set_float32_matmul_precision("medium")


def main():
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        max_epochs=config["train"]["num_epochs"],
        callbacks=[callbacks.PlottingCallback()],
        enable_checkpointing=False,
        logger=False,
    )

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
    # trainer.test(
    #     model=model,
    #     datamodule=datamodule,
    #     ckpt_path=_get_ckpt_path(),
    # )


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
