"""Module with WakeDataset class that handles dataset processing"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

import os
import json
from typing import Dict, Tuple


from torch import Tensor
from torchvision import io, transforms
import pytorch_lightning as pl
import torch

from src.visualization import plots


class WakeGANDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_config: Dict = config["data"]
        self.data_dir: Dict = {
            "train": os.path.join("data", "preprocessed", "tracked", "train"),
            "val": os.path.join("data", "preprocessed", "tracked", "val"),
            "test": os.path.join("data", "preprocessed", "tracked", "test"),
        }
        self.save: bool = config["models"]["save"]
        self.batch_size = config["train"]["batch_size"]
        self.num_workers = config["train"]["num_workers"]

    def setup(self, stage: str):
        self.dataset_train = WakeGANDataset(
            self.data_dir["train"],
            self.data_config,
            "train",
            save_norm_params=True if self.save else False,
        )
        self.dataset_val = WakeGANDataset(
            self.data_dir["val"],
            self.data_config,
            "val",
            norm_params=self.dataset_train.norm_params,
            save_norm_params=True if self.save else False,
        )
        self.dataset_test = WakeGANDataset(
            self.data_dir["test"],
            self.data_config,
            "test",
            norm_params=self.dataset_train.norm_params,
            save_norm_params=True if self.save else False,
        )

        plots.plot_histogram(self.dataset_train)
        plots.plot_histogram(self.dataset_val)
        plots.plot_histogram(self.dataset_test)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


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
        self.size = config["size"]
        self.clim = [config["figures"]["clim_ux"]]
        self.wt_grid = config["wt_grid"]

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
                transforms.Resize(self.size, antialias=True),
                transforms.Lambda(self._normalize_image),
            ]
        )

    def __getitem__(self, idx: int) -> tuple((Tensor, Tensor)):
        image, metadata = self._read_image(idx)
        image = self.transforms(image)

        inflow = image[:, :, 0]

        return image, inflow, metadata

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

    @staticmethod
    def rescale_back_to_velocity(tensor: Tensor, clim: list) -> Tensor:
        return tensor * (clim[0][1] - clim[0][0]) + clim[0][0]

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

    def _read_image(self, idx: int):
        image = torch.zeros((len(self), self.channels, self.size[0], self.size[1]))
        for i, fns in enumerate(self.images_fns):
            img_path = os.path.join(self.data_subdir[i], fns[idx])
            image = io.read_image(path=img_path, mode=io.ImageReadMode.GRAY)
            metadata = self._extract_metadata(fns[idx])
        return image, metadata

    def _extract_metadata(self, filename: str) -> Tuple:
        pos = int(filename.split("_")[0][6:])

        pos_x = pos // self.wt_grid[1]
        pos_y = pos % self.wt_grid[1]

        with open(os.path.join("data", "aux", "turns.json")) as f:
            angle_map = json.load(f)
        angle = angle_map[filename[4:6]]

        metadata = {
            "prec": float(filename[1:4]) / 100,
            "angle": angle,
            "pos": (pos_x, pos_y),
            "timestep": int(filename[-8:-7]),
        }

        return metadata

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
                with open(os.path.join("data", "aux", "norm_params.json"), "w") as f:
                    json.dump(norm_params, f)

        return norm_params

    def _calculate_statistics(self):
        images = torch.zeros((len(self), self.channels, self.size[0], self.size[1]))
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

    @staticmethod
    def transform_back_batch(
        images: Tensor, norm_type: "str", norm_params: Dict, clim: tuple
    ) -> Tensor:
        for im in images:
            im = WakeGANDataset.transform_back(im, norm_type, norm_params, clim)
        return images

    @staticmethod
    def transform_back(
        image: Tensor, norm_type: "str", norm_params: Dict, clim: tuple
    ) -> Tensor:
        image = WakeGANDataset.unnormalize_image(norm_type, norm_params, image)
        image = WakeGANDataset.rescale_back_to_velocity(image, clim)
        return image
