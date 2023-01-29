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

    trainer, dataset, dataloader = init(config)

    model = WakeGAN(config, dataset.norm_params)

    trainer.test(
        model=model,
        dataloaders=dataloader,
        ckpt_path=_get_ckpt_path(),
    )


def init(config: Dict):
    with open(os.path.join("data", "norm_params.json")) as f:
        norm_params = json.load(f)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        max_epochs=config["train"]["num_epochs"],
        callbacks=[callbacks.PlottingCallback()],
        enable_checkpointing=False,
        logger=False,
    )

    dataset_dev = dataset.WakeGANDataset(
        data_dir=os.path.join("data", "preprocessed", "tracked", "test"),
        config=config["data"],
        dataset_type="dev",
        norm_params=norm_params,
        save_norm_params=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset_dev,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        shuffle=False,
    )

    return trainer, dataset_dev, dataloader


def _get_ckpt_path():
    ckpt_path = os.path.join("logs", "lightning_logs")
    versions = os.listdir(ckpt_path)
    versions.sort()
    ckpt_name = os.listdir(os.path.join(ckpt_path, versions[-1], "checkpoints"))[0]
    ckpt_path = os.path.join(ckpt_path, versions[-1], "checkpoints", ckpt_name)
    return ckpt_path


if __name__ == "__main__":
    main()
