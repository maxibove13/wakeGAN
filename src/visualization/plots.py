"""Module with plotting functions"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

import os
from typing import Dict

from matplotlib import cm, pyplot as plt
import numpy as np
import torch

from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import MaxNLocator


class MetricsPlotter:
    def __init__(self, epochs: int, clim: tuple):
        self.epochs = epochs
        self.fig, self.axs = plt.subplots(3, 1, dpi=300)

        for ax in [self.axs[0], self.axs[1]]:
            ax.set(ylabel="loss")
            ax.xaxis.set_ticklabels([])

        self.axs[2].set(xlabel="epochs")
        self.axs[2].set(ylabel="RMSE [ms$^{-1}$]")
        sec_ax1 = self.axs[2].secondary_yaxis(
            "right",
            functions=(
                lambda x: x / clim[0][1] * 100,
                lambda x: x * clim[0][1] * 100,
            ),
        )
        sec_ax1.set_ylabel("RMSE [% of range]", rotation=270, labelpad=14)

        for i, ax in enumerate(self.axs):
            ax.set_xlim(1, self.epochs - 1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(visible=True)
        self.axs[1].set_ylim(0, 0.005)
        self.axs[2].set_ylim(0, 0.8)

    def plot(self, loss: Dict, rmse: Dict, epoch: int):
        """Plot losses and RMSE for train and dev sets"""
        x = np.arange(0, epoch + 1)

        self.axs[0].plot(
            x, loss["disc_synth"], label="Discriminator synth loss", color="C1"
        )
        self.axs[0].plot(
            x, loss["disc_real"], label="Discriminator real loss", color="C0"
        )

        self.axs[1].plot(
            x, loss["gen_adv"], label="Generator adversarial loss", color="r"
        )
        self.axs[1].plot(x, loss["gen_mse"], label="Generator MSE loss", color="C1")

        self.axs[2].plot(
            x, rmse["train"], label="RMSE Ux (Training)", color="g", ls="-"
        )
        self.axs[2].plot(x, rmse["dev"], label="RMSE Ux (Testing)", color="g", ls="--")

        if epoch == 0:
            for i, ax in enumerate(self.axs):
                ax.legend(
                    loc="upper right" if i == 1 else "upper right", fontsize="x-small"
                )

        self.fig.savefig(os.path.join("figures", "monitor", "metrics.png"))


class FlowImagePlotter:
    def __init__(
        self, channels: int, clim: tuple, monitor: bool = True, rmse: float = None
    ):

        self.monitor = monitor
        self.channels = channels
        self.clim = clim
        self.fig = {"img": plt.figure(dpi=300), "err": plt.figure(dpi=300)}

        self.grid_template = {
            "img": ImageGrid(
                self.fig["img"],
                111,
                nrows_ncols=(2, 3) if monitor else (2, 4),
                axes_pad=0.15 if monitor else (0.3, 0.2),
                share_all="True" if monitor else "False",
                label_mode="L" if monitor else "all",
                cbar_location="right",
                cbar_mode="edge",
                cbar_pad=None if monitor else 0.15,
            ),
            "err": ImageGrid(
                self.fig["err"],
                111,
                nrows_ncols=(2, 1) if monitor else (1, 4),
                axes_pad=0.15,
                share_all="True",
                cbar_location="right",
                cbar_mode="edge",
            ),
        }

        for c, ax in enumerate(self.grid_template["img"]):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            self._set_ax_addons_img(c, ax)

        for c, ax in enumerate(self.grid_template["err"]):
            self._set_ax_addons_err(c, ax)

        if not self.monitor:
            self.fig["img"].suptitle(
                f"Flow field comparison\n"
                f"Average RMSE for testing dataset: ({rmse:.3f})",
                y=0.9,
            )
            self.fig["err"].suptitle(
                f"Flow field error comparison\n"
                f"Average RMSE for testing dataset: ({rmse:.3f})",
                y=0.75,
            )

    def plot(self, images: list):
        if self.monitor:
            grid = {
                "img": [
                    images[0],
                    images[1],
                    images[1] - images[0],
                    images[2],
                    images[3],
                    images[3] - images[2],
                ],
                "err": [images[1] - images[0], images[3] - images[2]],
            }
        else:
            grid = {
                "img": images,
                "err": [
                    images[1] - images[0],
                    images[3] - images[2],
                    images[5] - images[4],
                    images[7] - images[6],
                ],
            }

        for c, (ax, im) in enumerate(zip(self.grid_template["img"], grid["img"])):
            im = ax.imshow(
                im,
                cmap=cm.coolwarm,
                interpolation="none",
                vmin=self.clim[0][0],
                vmax=self.clim[0][1],
            )
            if c == 2:
                cb = self.grid_template["img"].cbar_axes[0].colorbar(im)
                cb.set_label("[ms$^{-1}$]", rotation=270, labelpad=10)
            elif c == 3 and not self.monitor:
                cb = self.grid_template["img"].cbar_axes[0].colorbar(im)
                cb.ax.tick_params(labelsize=10)
                cb.set_label("[ms$^{-1}$]", rotation=270, labelpad=10, fontsize=10)
            elif c == 5:
                cb = self.grid_template["img"].cbar_axes[1].colorbar(im)
                cb.set_label("[ms$^{-1}$]", rotation=270, labelpad=10)
            elif c == 7 and not self.monitor:
                cb = self.grid_template["img"].cbar_axes[1].colorbar(im)
                cb.ax.tick_params(labelsize=10)
                cb.set_label("[ms$^{-1}$]", rotation=270, labelpad=10, fontsize=10)

        for c, (ax, im, cax) in enumerate(
            zip(
                self.grid_template["err"],
                grid["err"],
                self.grid_template["err"].cbar_axes,
            )
        ):

            err = ax.imshow(
                im,
                cmap=cm.coolwarm,
                interpolation="none",
                vmin=-1,
                vmax=1,
            )
            cbar = cax.colorbar(err)
            if not self.monitor:
                cbar = cax.colorbar(err)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("[ms$^{-1}$]", rotation=270, labelpad=10)

        fig_path = os.path.join("figures", "monitor" if self.monitor else "evaluation")

        self.fig["img"].savefig(
            os.path.join(fig_path, "images.png"), bbox_inches="tight"
        )
        self.fig["err"].savefig(
            os.path.join(fig_path, "pixel_diff.png"), bbox_inches="tight"
        )

    def _set_ax_addons_img(self, c, ax):

        if c == 0:
            ax.get_yaxis().set_visible(True)
            ax.set_yticks([])
        elif c == 1:
            ax.set_title("U$_{synth}$")

        if self.monitor:
            if c == 0:
                ax.set_ylabel("Training sample")
                ax.set_title("U$_{real}$")
            elif c == 2:
                ax.set_title("U$_{synth}$ - U$_{real}$")
            elif c == 3:
                ax.get_yaxis().set_visible(True)
                ax.set_ylabel("dev sample")
                ax.set_yticks([])
        else:
            if c == 0:
                ax.set_title("U$^{real}$")
                ax.set_ylabel(f"$Ux$ (Case #{c//2 + 1})", fontsize=10)
            elif c == 1:
                ax.set_ylabel(f"$Ux$ (Case #{c//2 + 1})", fontsize=10)
                ax.set_title("U$^{synth}$")
            elif c == 2:
                ax.set_title("U$^{real}$")
                ax.get_yaxis().set_visible(True)
                ax.set_ylabel(f"$Ux$ (Case #{c//2 + 1})", fontsize=10)
                ax.set_yticks([])
            elif c in [4, 6]:
                ax.get_yaxis().set_visible(True)
                ax.set_ylabel(f"$Ux$ (Case #{c//2 + 1})", fontsize=10)
                ax.set_yticks([])
                ax.set_aspect("equal")
            elif c == 3:
                ax.set_title("U$^{synth}$")
            elif c == 7:
                pass

    def _set_ax_addons_err(self, c, ax):
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        if self.monitor:
            if c == 0:
                ax.set_ylabel("training sample")
                ax.set_title("U$_{real}$")
                ax.set_yticks([])
                ax.set_title("U$_{fake}$ - U$_{real}$")
            elif c == 1:
                ax.set_ylabel("Testing sample")
        else:
            if c == 0:
                ax.set_ylabel("$U_x^{fake} - U_x^{real}$", fontsize=10)
            ax.set_title(f"Case #{c+1}")


class ProfilesPlotter:
    def __init__(self, wt_d: float, limits: tuple, size: tuple, metadata: dict):
        self.fig = plt.figure(figsize=(10, 25), dpi=300)
        self.wt_d = wt_d
        self.grid = ImageGrid(
            self.fig,
            111,
            nrows_ncols=(4, 7),
            axes_pad=(0.15, 0.70),
            share_all="False",
            aspect=False,
            cbar_mode=None,
        )

        x_left = -self.wt_d * limits[0]
        x_right = self.wt_d * limits[1]
        y_bottom = -self.wt_d * limits[2]
        y_top = self.wt_d * limits[3]

        self.x = np.linspace(x_left, x_right, num=size[0])
        self.y = np.linspace(y_bottom, y_top, num=size[0])

        self.prec = [m["prec"] for m in metadata]
        self.angle = [m["angle"] for m in metadata]
        self.pos = [m["pos"] for m in metadata]

    def plot(self, images: list):

        # iterate over grid rows
        for i, ax_row in enumerate(self.grid.axes_row):
            im_real = images[2 * i]
            im_synth = images[2 * i + 1]
            # iterate over grid columns
            for j, ax in enumerate(ax_row):
                x_index = np.argmin(abs((j - 2) * self.wt_d - self.x))

                real_prof = im_real[:, x_index].cpu()
                synth_prof = im_synth[:, x_index].cpu()

                (real_curve,) = ax.plot(
                    real_prof, self.y / self.wt_d, c="k", ls="-", label="CFD"
                )
                (synth_curve,) = ax.plot(
                    synth_prof, self.y / self.wt_d, c="r", ls="-.", label="GAN"
                )
                ax.set_xlim(left=1, right=12)
                ax.grid()
                ax.set_title(f"{j-2}D")
                ax.tick_params(direction="in")
                if i == len(self.grid.axes_row) - 1 and j == 0:
                    ax.set_xlabel("$U_x$ [ms$^{-1}$]", fontsize=16)
                if j == 0:
                    m_s = "ms$^{-1}$"
                    ax.set_ylabel(
                        f"$y/D$ -#{i+1}, prec.: {self.prec[i]} {m_s}, {self.angle[i]}°, {self.pos[i]}",
                        fontsize=14,
                    )

        self.fig.legend(
            handles=[real_curve, synth_curve],
            bbox_to_anchor=(0.0, 0.0, 0.9, 0.097),
            ncol=2,
            fontsize="xx-large",
        )
        self.fig.savefig(
            os.path.join("figures", "evaluation", "profiles.png"),
            dpi=300,
            bbox_inches="tight",
        )


def plot_histogram(dataset):
    """
    Plot histogram of an image dataset (normalize and unnormalized) in order to see the distribution of pixel values
    """

    images_unnorm = torch.zeros(
        (len(dataset), dataset.channels, dataset.size[0], dataset.size[1])
    )
    images_norm = images_unnorm.clone()
    for c, (image, _, _) in enumerate(dataset):
        images_norm[c] = image
        images_unnorm[c] = dataset.unnormalize_image(
            dataset.norm_type, dataset.norm_params, image
        )

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
