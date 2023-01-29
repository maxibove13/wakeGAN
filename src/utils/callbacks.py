"""Module with Pytorch Lightning callbacks"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "01/23"

import os
import logging

import torch
from pytorch_lightning import callbacks

from src.visualization import plots
from src.data import dataset


class LoggingCallback(callbacks.Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_train_start(self, trainer, pl_module):
        self.logger.info(f"Training {pl_module.__class__.__name__}:\n")

        if torch.cuda.is_available():
            self.logger.info(
                f"Using device: {torch.cuda.get_device_name()}"
                f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB\n"
            )
        else:
            self.logger.info("Using device: CPU\n")

        self.logger.info(
            f"Initialized Generator with {sum(p.numel() for p in pl_module.generator.parameters())} params\n"
            f"Initialized Discriminator with {sum(p.numel() for p in pl_module.discriminator.parameters())} params\n"
        )

        self.logger.info(
            f"{pl_module.loss['bce']}\n"
            f"Adam optimizer for generator with initial lr: {pl_module.lr} and betas {pl_module.betas}\n"
            f"Adam optimizer for discriminator with initial lr: {pl_module.lr} and betas: {pl_module.betas}\n"
        )

        self.logger.info(
            f"Training set samples: {len(trainer.datamodule.dataset_train)}\n"
            f"Testing set samples: {len(trainer.datamodule.dataset_dev)}\n"
            f"Number of epochs: {trainer.max_epochs}\n"
            f"Mini batch size: {pl_module.minibatch_size}\n"
            f"Number of training batches: {len(trainer.train_dataloader)}\n"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        self.logger.info(
            f"{trainer.current_epoch + 1:03d}/{trainer.max_epochs}, "
            f"loss disc: {trainer.callback_metrics['d_loss']:.2f} / "
            f"loss gen: {trainer.callback_metrics['g_loss']:.2f}, "
            f"rmse train: {trainer.callback_metrics['rmse_train_epoch']:.2f} / rmse dev: {trainer.callback_metrics['rmse_dev_epoch']:.2f} "
        )


class PlottingCallback(callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        self.plotters = {
            "metrics": plots.MetricsPlotter(trainer.max_epochs, pl_module.clim),
            "flow": plots.FlowImagePlotter(pl_module.channels, pl_module.clim),
        }

    def on_train_batch_end(
        trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        self.plotters["flow"].plot(
            [
                pl_module.images["real"][0].detach().cpu(),
                pl_module.images["synth"][0].detach().cpu(),
                pl_module.images_dev["real"][0].detach().cpu(),
                pl_module.images_dev["synth"][0].detach().cpu(),
            ]
        )

    def on_train_end(self, trainer, pl_module):
        pass
