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
from src.utils import utils


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

        if not torch.cuda.is_available():
            self.device = "cpu"
            device_name = "CPU"
        else:
            if torch.cuda.device_count() == 1:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device(self.device_name)
            device_name = torch.cuda.get_device_name()

        logger.info(
            f"Using device: {device_name}"
            f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB\n"
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
            f"Using {self.criterion}\n"
            f"Using Adam optimizer for generator with lr: {self.lr} and betas {self.betas}\n"
            f"Using Adam optimizer for discriminator with lr: {self.lr} and betas: {self.betas}\n"
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

        self.generator.running_loss = 0
        self.discriminator.running_loss = 0

        dataloader = {
            "train": self.dataset["train"].get_dataloader(
                batch_size=self.minibatch_size, num_workers=self.workers
            ),
            "dev": self.dataset["dev"].get_dataloader(
                batch_size=self.minibatch_size, num_workers=self.workers
            ),
        }

        logger.info(
            f"Training set samples: {len(self.dataset['train'])}\n"
            f"Testing set samples: {len(self.dataset['dev'])}\n"
            f"Number of epochs: {self.epochs}\n"
            f"Mini batch size: {self.minibatch_size}\n"
            f"Number of training batches: {len(dataloader['train'])}\n"
        )

        self._training_loop(dataloader)

    def _training_loop(self, dataloader: Dict):
        for epoch in range(self.epochs):
            running_mse = 0
            for c, (images, inflows) in enumerate(dataloader["train"]):
                images = images.float().to(self.device)
                inflows = inflows.float().to(self.device)
                synths = self.generator(inflows)

                self._train_discriminator(inflows, images, synths)
                self._train_generator(synths, inflows)

                running_mse += self._calculate_batch_mse(images, synths)

            self.discriminator.loss.append(
                self.discriminator.running_loss / len(dataloader["train"])
            )
            self.generator.loss.append(
                self.generator.running_loss / len(dataloader["train"])
            )

            mse = running_mse / len(dataloader["train"])
            rmse = torch.sqrt(mse)

            logger.info(
                f"{epoch + 1:03d}/{self.epochs}, "
                f"loss disc: {self.discriminator.loss[-1]:.2f}, "
                f"loss gen: {self.generator.loss[-1]:.2f}, "
                f"rmse: {rmse:.2f}, "
            )

    # def _evaluate_model()

    def _calculate_batch_mse(self, images, synths):
        for (img, synth) in zip(images.detach(), synths.detach()):
            img = self.dataset["train"].unnormalize_image(img)
            synth = self.dataset["train"].unnormalize_image(synth)

            img = self.dataset["train"].rescale_back_to_velocity(img)
            synth = self.dataset["train"].rescale_back_to_velocity(synth)

            mse = utils.calculate_mse(img, synth, np.prod(self.size))
        mse /= len(images)
        return mse

    def _train_discriminator(self, inflows, images, synths) -> float:
        """Train Discriminator: max log(D(x)) + log(1 - D(G(z)))"""

        pred_real = self.discriminator(images, inflows)
        pred_synth = self.discriminator(synths, inflows)

        loss_real = self.criterion(pred_real, torch.ones_like(pred_real))
        loss_synth = self.criterion(pred_synth, torch.zeros_like(pred_synth))

        loss = (loss_real + loss_synth) / 2

        self.optimizer["discriminator"].zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer["discriminator"].step()

        self.discriminator.running_loss += loss.item()

    def _train_generator(self, synths, inflows) -> float:
        """Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))"""

        pred_synth = self.discriminator(synths, inflows)

        loss = self.criterion(pred_synth, torch.ones_like(pred_synth))

        self.optimizer["generator"].zero_grad()
        loss.backward()
        self.optimizer["generator"].step()

        self.generator.running_loss += loss.item()

    def plot_monitor_figures():
        ...

    def save_models():
        ...

    def _load_model(
        self, model: torch.nn.Module, name: str, optimizer: torch.optim.Optimizer
    ) -> None:
        checkpoint = torch.load(os.path.join("models", name), map_location=self.device)

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr

        logger.info(f"Loaded model {name} from disk")
