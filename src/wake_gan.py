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
from src.visualization import plots


class WakeGAN:
    def __init__(self, config: Dict):
        self.channels: int = config["data"]["channels"]
        self.data_dir: Dict = {
            "train": os.path.join("data", "preprocessed", "tracked", "train"),
            "test": os.path.join("data", "preprocessed", "tracked", "test"),
        }
        self.data_config: Dict = config["data"]

        self.device_name: str = config["train"]["device"]
        self.minibatch_size: int = config["train"]["batch_size"]
        self.epochs: int = config["train"]["num_epochs"]

    def set_device(self):
        if torch.cuda.is_available():
            self.device = "cpu"
        else:
            if torch.cuda.device_count() == 1:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device(self.device_name)

        logger.info(
            f"\n"
            f"Using device: {torch.cuda.get_device_name()} "
            f"with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB"
        )

    def preprocess_dataset(self):

        dataset_train = WakeGANDataset(
            data_dir=self.data_dir["train"],
            config=self.data_config,
        )

        plots.plot_histogram(dataset_train)

    def initialize_models():
        ...

    def define_loss_and_optimizer():
        ...

    def load_pretrained_models():
        ...

    def train_discriminator():
        ...

    def train_generator():
        ...

    def evaluate_model():
        ...

    def plot_monitor_figures():
        ...

    def save_models():
        ...

    def rescale_back_to_velocity(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * (self.clim_ux[1] - self.clim_ux[0]) + self.clim_ux[0]
