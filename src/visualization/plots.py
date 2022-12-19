"""Module with plotting functions"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

import os

from matplotlib import pyplot as plt
import numpy as np
import torch

from src.data.dataset import WakeGANDataset


def plot_histogram(dataset: WakeGANDataset):
    """
    Plot histogram of an image dataset (normalize and unnormalized) in order to see the distribution of pixel values
    """

    images_unnorm = torch.zeros(
        (len(dataset), dataset.channels, dataset.size[0], dataset.size[1])
    )
    images_norm = images_unnorm.clone()
    for c, (image, inflow) in enumerate(dataset):
        images_norm[c] = image
        images_unnorm[c] = dataset.unnormalize_image(image)

    fig, axs = plt.subplots(1, 2)
    for c, images in enumerate([images_unnorm, images_norm]):
        images_np = images.numpy()
        flat_images = images_np.flatten()
        mean = flat_images.mean()
        std = flat_images.std()

        axs[c].hist(flat_images, bins=100, density=True)

        x, y = compute_gaussian_curve(
            mean, std, limits=[np.min(flat_images), np.max(flat_images)]
        )
        axs[c].plot(x, y, "r")
        axs[c].axvline(mean, color="k", linestyle="dashed", linewidth=1)
        axs[c].axvline(mean + std, color="k", linestyle="dashed", linewidth=1)
        axs[c].axvline(mean - std, color="k", linestyle="dashed", linewidth=1)

        axs[c].set_title(f"$\mu={mean:.2f}$, $\sigma={std:.2f}$")
        axs[c].set_xlabel(
            f'Pixel value {"(unnormalized)" if c == 0 else "(normalized)"}'
        )
        if c == 0:
            axs[c].set_ylabel("Probability density")

        figname = (
            "hist_pixel_values_train_set.png"
            if dataset.type == "train"
            else "hist_pixel_values_dev_set.png"
        )

    norm_type = (
        f"to {dataset.range} range"
        if dataset.norm_type == "min_max"
        else f"by mean and std"
    )
    fig.suptitle(f"Histogram of pixel values: normalization {norm_type}")
    fig.savefig(os.path.join("figures", "reference", figname), dpi=300)


def compute_gaussian_curve(mean, std, limits):
    """Compute gaussian curve for a given mean and std"""
    x = np.linspace(min(limits), max(limits), 100)
    y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * std**2))
    return x, y
