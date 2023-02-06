#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to generate an entire WF from the firsts inflows"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

import os
import json
import yaml

import numpy as np
from scipy.io import loadmat
from scipy import interpolate
from matplotlib import pyplot as plt, cm
import pytorch_lightning as pl
import torch

from src.wakegan import WakeGAN
from src.data.dataset import WakeGANDataset

with open("config.yaml") as file:
    config = yaml.safe_load(file)

with open(os.path.join("data", "aux", "norm_params.json")) as file:
    norm_params = json.load(file)

with open(os.path.join("data", "aux", "turns.json")) as f:
    turns = json.load(f)

wt_coord = loadmat(os.path.join("data", "aux", "coord_layout.mat"))
wt_xy = {}

for idx, turn in enumerate(turns.keys()):
    wt_xy[turn] = wt_coord["coord_layout"][:, 0:2, idx]

path = os.path.join("data", "generated", "testing", "real")

size = (64, 64)
wt_d = 126


wf_grid = (
    np.load(os.path.join("data", "aux", "grid_x.npy")),
    np.load(os.path.join("data", "aux", "grid_y.npy")),
)

x_sample = np.linspace(-wt_d * 2, wt_d * 4, num=size[0])
y_sample = np.linspace(-wt_d * 2, wt_d * 2, num=size[1])


def main():

    dataset = WakeGANDataset(
        os.path.join("data", "preprocessed", "tracked", "test"),
        config["data"],
        "test",
        norm_params=norm_params,
    )
    wakegan = WakeGAN.load_from_checkpoint(
        _get_ckpt_path(), config=config, norm_params=norm_params
    )
    wakegan.eval()

    for c, (prec, angle, turn, timestep, clim) in enumerate(
        [
            ["5.76", "-5.0", "n5", "0", (1, 6)],
            ["7.68", "0.0", "pr", "0", (3, 8)],
        ]
    ):

        fig, axs = plt.subplots(2, 1)
        fig_err, axs_err = plt.subplots(1, 1)

        wf_values_synth = np.full((wf_grid[0].shape[0], wf_grid[1].shape[0]), np.nan)
        wf_values_real = np.full((wf_grid[0].shape[0], wf_grid[1].shape[0]), np.nan)
        for row in range(3):
            for col in range(5):
                if config["data"]["t_window"] == 1000:
                    filename = f"{prec}_{angle}_({2-row},{col})_{timestep}.pt"
                elif config["data"]["t_window"] == 4000:
                    filename = f"{prec}_{angle}_({2-row},{col}).pt"
                else:
                    raise ValueError("Invalid t_window value")
                real = torch.load(os.path.join(path, filename))  # [-1, 1]
                if col == 0:
                    inflow = real[:, :, 0].unsqueeze(dim=0)  # 1x1x64, NCH
                    synth = wakegan(inflow.to(wakegan.device))  # 1x1x64x64, NCHW

                wt_idx = row * 5 + col
                wt_pos = (wt_xy[turn][wt_idx][0], wt_xy[turn][wt_idx][1])
                for ax in axs:
                    ax.plot(wt_pos[0], wt_pos[1], "kx", markeredgewidth=1, markersize=4)
                axs_err.plot(
                    wt_pos[0], wt_pos[1], "kx", markeredgewidth=1, markersize=4
                )

                sample_points = (x_sample + wt_pos[0], y_sample + wt_pos[1])

                if col > 0:
                    x = np.repeat(sample_points[0][0], size[0])
                    y = np.linspace(sample_points[1][0], sample_points[1][-1], size[0])
                    X, Y = np.meshgrid(x, y)
                    inflow = extrap((X, Y)).astype("float32")[:, 0]
                    inflow = torch.from_numpy(inflow).unsqueeze(dim=0).unsqueeze(dim=0)
                    synth = wakegan(inflow.to(wakegan.device))  # 1x1x64x64, NCHW
                # extrapolator of synth
                extrap = interpolate.RegularGridInterpolator(
                    sample_points,
                    synth.detach().squeeze().cpu().numpy().T,
                    method="nearest",
                    bounds_error=False,
                    fill_value=None,
                )
                # unnormalize
                synth = dataset.transform_back(synth, "min_max", norm_params, [[0, 12]])
                real = dataset.transform_back(real, "min_max", norm_params, [[0, 12]])

                # interpolate synth to wf_grid
                interp_to_wf_synth = interpolate.RegularGridInterpolator(
                    sample_points,
                    synth.detach().squeeze().cpu().numpy().T,
                    bounds_error=False,
                    fill_value=None,
                )
                interp_to_wf_real = interpolate.RegularGridInterpolator(
                    sample_points,
                    real.detach().squeeze().cpu().numpy().T,
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )
                ix_0 = np.abs(wf_grid[0] - sample_points[0][0]).argmin()
                ix_1 = np.abs(wf_grid[0] - sample_points[0][-1]).argmin()
                iy_0 = np.abs(wf_grid[1] - sample_points[1][0]).argmin()
                iy_1 = np.abs(wf_grid[1] - sample_points[1][-1]).argmin()

                X, Y = np.meshgrid(wf_grid[0][ix_0:ix_1], wf_grid[1][iy_0:iy_1])

                wf_values_synth[ix_0:ix_1, iy_0:iy_1] = interp_to_wf_synth((X, Y)).T
                wf_values_real[ix_0:ix_1, iy_0:iy_1] = interp_to_wf_real((X, Y)).T

        lims = [wf_grid[0][0], wf_grid[0][-1], wf_grid[1][0], wf_grid[1][-1]]

        for i, (ax, image) in enumerate(
            zip(axs, [wf_values_real.T, wf_values_synth.T])
        ):
            im = ax.imshow(
                image,
                cmap=cm.coolwarm,
                extent=lims,
                origin="lower",
                vmin=clim[0],
                vmax=clim[1],
            )
            cbar = fig.colorbar(
                im, ax=ax, orientation="vertical", fraction=0.0237, pad=0.04
            )
            cbar.set_label("[ms$^{-1}$]", labelpad=12, rotation=270)
            ax.set_xlim(0, 4500)
            ax.set_ylabel("$y$ [m]")
            if i == 1:
                ax.set_title("synthetic wind farm $Ux$ flow", fontsize=10)
                ax.set_xlabel("$x$ [m]")
            else:
                ax.set_title("real wind farm $Ux$ flow", fontsize=10)
                ax.set_xticklabels([])

        im_err = axs_err.imshow(
            (wf_values_synth.T - wf_values_real.T),
            cmap=cm.coolwarm,
            extent=lims,
            origin="lower",
            vmin=-1,
            vmax=1,
        )
        cbar = fig.colorbar(
            im_err, ax=axs_err, orientation="vertical", fraction=0.0237, pad=0.04
        )
        cbar.set_label("ms$^{-1}$]", labelpad=12, rotation=270)
        axs_err.set_xlim(0, 4500)
        axs_err.set_ylabel("$y$ [m]")
        axs_err.set_title("U$^{real}$ - U$^{synth}$ wind farm $Ux$ flow", fontsize=10)

        fig.savefig(
            f"wf_flow_{prec}_{angle}_{timestep}.png", bbox_inches="tight", dpi=1000
        )
        fig_err.savefig(
            f"wf_err_{prec}_{angle}_{timestep}.png", bbox_inches="tight", dpi=1000
        )


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
