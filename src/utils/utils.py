"""Module with utils functions"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

from torch import Tensor, sqrt, sum


def calculate_mse(real: Tensor, pred: Tensor, n: int) -> Tensor:
    return sum((pred - real) ** 2) / n
