"""Module with WakeDataset class that handles dataset processing"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

import os
import json
from typing import Dict, Tuple


import torch
from torch import Tensor
from torchvision import transforms
from torchvision import io


class WakeGANDataset:
    def __init__(
        self,
        data_dir: str,
        config: Dict,
        dataset_type: str,
        norm_params: Dict = None,
        save_norm_params: bool = False,
    ):

        self.type = dataset_type
        self.data_subdir = [os.path.join(data_dir, "ux")]
        self.channels = config["channels"]
        self.original_size = config["original_size"]
        self.size = config["size"]
        self.clim = [config["figures"]["clim_ux"]]

        self.norm_type = config["normalization"]["type"]
        self.range = config["normalization"]["range"]
        self.user_mean = config["normalization"]["mean_std"][0]
        self.user_std = config["normalization"]["mean_std"][1]

        self.images_fns = []
        self.images_fns += [list(os.listdir(self.data_subdir[0]))]

        if self.channels == 2:
            self.data_subdir.append(os.path.join(data_dir, "uy"))
            self.images_fns.append(list(os.listdir(self.data_subdir[1])))
            self.clim.append(config["figures"]["clim_uy"])

        self.mean, self.std, self.min, self.max = self._calculate_statistics()

        self.norm_params = self._set_norm_params(norm_params, save_norm_params)

        self.transforms = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float),
                transforms.Resize(self.size),
                transforms.Lambda(self._normalize_image),
            ]
        )

    def __getitem__(self, idx: int) -> tuple((Tensor, Tensor)):

        image = self._cat_grayscale_images_as_channels(idx)
        image = self.transforms(image)

        inflow = image[:, :, 0]

        return image, inflow

    def __len__(self) -> int:
        return len(self.images_fns[0])

    @staticmethod
    def unnormalize_image(norm_type: str, norm_params: Dict, image: Tensor):
        if norm_type == "min_max":
            image = WakeGANDataset._rescale_back_to_original_range(
                image,
                norm_params["range"],
                norm_params["max"],
                norm_params["min"],
            )
        elif norm_type == "z_score":
            mean = norm_params["mean"]
            std = norm_params["std"]
            unnormalize = transforms.Normalize((-mean / std), (1.0 / std))
            image = unnormalize(image)
        else:
            raise ValueError(f"Normalization type {norm_type} not supported")

        return image

    def _normalize_image(self, image: Tensor):
        if self.norm_type == "min_max":
            image = self._rescale_to_range(
                image,
                self.norm_params["range"],
                self.norm_params["max"],
                self.norm_params["min"],
            )
        elif self.norm_type == "z_score":
            normalize_z_score = transforms.Normalize(
                self.norm_params["mean"], self.norm_params["std"]
            )
            image = normalize_z_score(image)
        else:
            raise ValueError(f"Normalization type {self.norm_type} not supported")

        return image

    def _cat_grayscale_images_as_channels(self, idx: int):
        image = torch.zeros(
            (len(self), self.channels, self.original_size[0], self.original_size[1])
        )
        for i, fns in enumerate(self.images_fns):
            img_path = os.path.join(self.data_subdir[i], fns[idx])
            image = io.read_image(path=img_path, mode=io.ImageReadMode.GRAY)
        return image

    def _set_norm_params(self, norm_params, save: bool = False):
        if not norm_params:
            if self.norm_type == "min_max":
                norm_params = {
                    "range": self.range,
                    "min": self.min.item(),
                    "max": self.max.item(),
                }
            elif self.norm_type == "z_score":
                norm_params = {
                    "mean": self.user_mean if self.user_mean else self.mean.item(),
                    "std": self.user_std if self.user_std else self.std.item(),
                }
            else:
                raise ValueError(f"Normalization type {self.norm_type} not supported")
            if save:
                with open(os.path.join("data", "norm_params.json"), "w") as f:
                    json.dump(norm_params, f)

        return norm_params

    def _calculate_statistics(self):
        images = torch.zeros(
            (len(self), self.channels, self.original_size[0], self.original_size[1])
        )
        for i, fns in enumerate(self.images_fns):
            for c, image_fn in enumerate(fns):
                img_path = os.path.join(self.data_subdir[i], image_fn)
                image = io.read_image(path=img_path, mode=io.ImageReadMode.GRAY)
                image = transforms.ConvertImageDtype(torch.float)(image)
                images[c] = image

        return (
            torch.mean(images),
            torch.std(images),
            torch.min(images),
            torch.max(images),
        )

    def set_loader(self, batch_size: int, num_workers: int, shuffle: bool = True):
        self.loader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
        )

    @staticmethod
    def rescale_back_to_velocity(tensor: Tensor, clim: list) -> Tensor:
        return tensor * (clim[0][1] - clim[0][0]) + clim[0][0]

    @staticmethod
    def _rescale_to_range(
        x: Tensor, range: tuple, x_max: float, x_min: float
    ) -> Tensor:
        a, b = range
        return (b - a) * (x - x_min) / (x_max - x_min) + a

    @staticmethod
    def _rescale_back_to_original_range(
        x: Tensor, range: tuple, x_max: float, x_min: float
    ) -> Tensor:
        a, b = range
        return (x - a) * (x_max - x_min) / (b - a) + x_min
