"""Module with utils functions"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

from typing import Dict

from torch import Tensor, sqrt, sum
import numpy as np

from src.data.dataset import WakeGANDataset


def calculate_mse(real: Tensor, pred: Tensor, n: int) -> Tensor:
    return sum((pred - real) ** 2) / n


def calculate_batch_mse(images: Tensor, synths: Tensor, norm: Dict) -> Tensor:
    mse = 0
    for (img, synth) in zip(images, synths):
        img = WakeGANDataset.transform_back(img, norm["type"], norm["params"])
        synth = WakeGANDataset.transform_back(synth, norm["type"], norm["params"])

        mse += calculate_mse(
            img.flatten(), synth.flatten(), img.shape[0] * img.shape[1]
        )
    mse /= len(images)
    return mse
