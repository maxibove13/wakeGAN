"""Module with WakeGAN class that defines general methods for training the network"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

import os
import time
from typing import List, Dict

from matplotlib import pyplot as plt
import numpy as np
import torch

from src.data.dataset import WakeGANDataset
from src.utils.logger import logger
from src.models import dcgan
from src.visualization import plots


class WakeGAN:
    def __init__(self, config: Dict):
        self.channels: int = config["data"]["channels"]
        self.size: tuple = config["data"]["size"]
        self.data_dir: Dict = {
            "train": os.path.join("data", "preprocessed", "tracked", "train"),
            "test": os.path.join("data", "preprocessed", "tracked", "test"),
        }
        self.data_config: Dict = config["data"]

        self.device_name: str = config["train"]["device"]
        self.lr: float = config["train"]["lr"]
        self.minibatch_size: int = config["train"]["batch_size"]
        self.epochs: int = config["train"]["num_epochs"]
        self.betas: tuple = (0.5, 0.999)
        self.workers: int = config["train"]["num_workers"]

        self.load = config["models"]["load"]
        self.save = config["models"]["save"]

        self.net_name = {}
        self.net_name["generator"] = config["models"]["name_gen"]
        self.net_name["discriminator"] = config["models"]["name_disc"]

        self.feat_gen = config["models"]["f_g"]
        self.feat_disc = config["models"]["f_d"]

    def set_device(self) -> None:
        if torch.cuda.is_available():
            self.device = "cpu"
        else:
            if torch.cuda.device_count() == 1:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device(self.device_name)

        logger.info(
            f"\n"
            f"Using device: {torch.cuda.get_device_name()}"
            f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB"
        )

    def preprocess_dataset(self) -> None:
        self.dataset = {}
        self.dataset["train"] = WakeGANDataset(
            data_dir=self.data_dir["train"],
            config=self.data_config,
            dataset_type="train",
        )
        self.dataset["dev"] = WakeGANDataset(
            data_dir=self.data_dir["test"],
            config=self.data_config,
            dataset_type="dev",
            norm_params=self.dataset["train"].norm_params,
        )

        plots.plot_histogram(self.dataset["train"])
        plots.plot_histogram(self.dataset["dev"])

    def initialize_models(self) -> None:
        self.generator = dcgan.Generator(
            self.channels,
            self.size[0],
            self.feat_gen,
        ).to(self.device)

        self.discriminator = dcgan.Discriminator(
            self.channels,
            self.feat_disc,
            self.size[0],
        ).to(self.device)

        logger.info(
            f"Initializing Generator with {sum(p.numel() for p in self.generator.parameters())} params\n"
            f"Initializing Discriminator with {sum(p.numel() for p in self.discriminator.parameters())} params\n"
        )

    def define_loss_and_optimizer(self) -> None:
        self.criterion = torch.nn.BCELoss()

        self.optimizer = {}
        self.optimizer["generator"] = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=self.betas
        )
        self.optimizer["discriminator"] = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=self.betas
        )

        logger.info(
            f"Using {self.criterion} loss function\n"
            f"Using Adam optimizer for generator with learning rate: {self.lr} and betas {self.betas}\n"
            f"Using Adam optimizer for discriminator with learning rate: {self.lr} and betas: {self.betas}\n"
        )

    def load_pretrained_models(self) -> None:
        self._load_model(
            self.generator, self.net_name["generator"], self.optimizer["generator"]
        )
        self._load_model(
            self.discriminator,
            self.net_name["discriminator"],
            self.optimizer["discriminator"],
        )

    def train(self):
        self.generator.train()
        self.discriminator.train()

        dataloader = {
            "train": self.dataset["train"].get_dataloader(
                batch_size=self.minibatch_size, num_workers=self.workers
            ),
            "dev": self.dataset["dev"].get_dataloader(
                batch_size=self.minibatch_size, num_workers=self.workers
            ),
        }

        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            for c, (images, inflows) in enumerate(dataloader["train"]):
                images = images.float().to(self.device)
                inflows = inflows.float().to(self.device)
                synths = self.generator(inflows)

                self.train_discriminator(inflows, images, synths)
                self.train_generator(synths, inflows)
                logger.info(f"     {c}")

    def train_discriminator(self, inflows, images, synths):
        """Train Discriminator: max log(D(x)) + log(1 - D(G(z)))"""

        pred_real = self.discriminator(images, inflows)
        pred_synth = self.discriminator(synths, inflows)

        loss_real = self.criterion(pred_real, torch.ones_like(pred_real))
        loss_synth = self.criterion(pred_synth, torch.zeros_like(pred_synth))

        loss = (loss_real + loss_synth) / 2

        self.optimizer["discriminator"].zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer["discriminator"].step()

    def train_generator(self, synths, inflows):
        """Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))"""

        pred_synth = self.discriminator(synths, inflows)

        loss = self.criterion(pred_synth, torch.ones_like(pred_synth))

        self.optimizer["generator"].zero_grad()
        loss.backward()
        self.optimizer["generator"].step()

    def evaluate_model():
        ...

    def plot_monitor_figures():
        ...

    def save_models():
        ...

    def rescale_back_to_velocity(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * (self.clim_ux[1] - self.clim_ux[0]) + self.clim_ux[0]

    def _load_model(
        self, model: torch.nn.Module, name: str, optimizer: torch.optim.Optimizer
    ) -> None:
        checkpoint = torch.load(os.path.join("models", name), map_location=self.device)

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr

        logger.info(f"Loaded model {name} from disk")
