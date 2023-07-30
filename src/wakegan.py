"""Module with WakeGAN class that defines general methods for training the network"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

from typing import Dict
import json
import os

from torchvision import transforms
import pytorch_lightning as pl
import torch
import torchmetrics
import yaml

from src.data.dataset import WakeGANDataset
from src.models import dcgan


class WakeGAN(pl.LightningModule):
    def __init__(self, config: Dict = None, norm_params: Dict = None):
        super().__init__()
        if not config:
            with open("config.yaml") as file:
                config = yaml.safe_load(file)

        self._set_hparams(config, norm_params)

        torch.manual_seed(0)

        self.generator = dcgan.Generator(self.channels, self.size[0], self.feat_gen)
        self.discriminator = dcgan.Discriminator(
            self.channels, self.feat_disc, self.size[0]
        )

        self.loss = {
            "bce": torch.nn.BCELoss(),
            "mse": torch.nn.MSELoss(),
        }
        self.mse_train = torchmetrics.MeanSquaredError(squared=False)
        self.mse_val = torchmetrics.MeanSquaredError(squared=False)
        self.mse_test = torchmetrics.MeanSquaredError(squared=False)

        self.fid_train = torchmetrics.image.fid.FrechetInceptionDistance(
            feature=64, normalize=False
        )
        self.fid_val = torchmetrics.image.fid.FrechetInceptionDistance(
            feature=64, normalize=False
        )
        self.fid_test = torchmetrics.image.fid.FrechetInceptionDistance(
            feature=64, normalize=False
        )

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        reals, inflows, _ = batch

        synths = self(inflows)

        self.fid_train.update(
            transforms.ConvertImageDtype(dtype=torch.uint8)(reals).repeat(1, 3, 1, 1),
            real=True,
        )
        self.fid_train.update(
            transforms.ConvertImageDtype(dtype=torch.uint8)(synths).repeat(1, 3, 1, 1),
            real=False,
        )

        if optimizer_idx == 0:
            loss = self._train_discriminator(inflows, synths, reals)

        if optimizer_idx == 1:
            loss = self._train_generator(inflows, synths, reals)

        self.images_train = {
            "real": WakeGANDataset.transform_back(
                reals.clone().squeeze(),
                self.norm["type"],
                self.norm["params"],
                self.clim,
            ),
            "synth": WakeGANDataset.transform_back(
                synths.clone().squeeze(),
                self.norm["type"],
                self.norm["params"],
                self.clim,
            ),
        }

        self.mse_train(
            self.images_train["real"],
            self.images_train["synth"],
        )

        self.log(
            "rmse_train_epoch",
            self.mse_train,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "fid_train_epoch",
            self.fid_train.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "reals": reals, "synths": synths}

    def validation_step(self, batch, batch_idx):
        reals, inflows, metadatas = batch
        synths = self(inflows)

        # update FID, image expected to be [N, 3, H, W], dtype=uint8 if normalize=False
        self.fid_val.update(
            transforms.ConvertImageDtype(dtype=torch.uint8)(reals).repeat(1, 3, 1, 1),
            real=True,
        )
        self.fid_val.update(
            transforms.ConvertImageDtype(dtype=torch.uint8)(synths).repeat(1, 3, 1, 1),
            real=False,
        )

        self.images_val = {
            "real": WakeGANDataset.transform_back(
                reals.squeeze(), self.norm["type"], self.norm["params"], self.clim
            ),
            "synth": WakeGANDataset.transform_back(
                synths.squeeze(), self.norm["type"], self.norm["params"], self.clim
            ),
        }

        self.metadatas_val = metadatas

        self.mse_val(
            self.images_val["real"],
            self.images_val["synth"],
        )

        self.log(
            "rmse_val_epoch",
            self.mse_val,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "fid_val_epoch",
            self.fid_val.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {
            "reals": reals,
            "synths": synths,
            "metadatas": metadatas,
        }

    def test_step(self, batch, batch_idx):
        reals, inflows, metadatas = batch
        synths = self(inflows)

        self.fid_test.update(
            transforms.ConvertImageDtype(dtype=torch.uint8)(reals).repeat(1, 3, 1, 1),
            real=True,
        )
        self.fid_test.update(
            transforms.ConvertImageDtype(dtype=torch.uint8)(synths).repeat(1, 3, 1, 1),
            real=False,
        )

        self.images_test = {
            "real": WakeGANDataset.transform_back(
                reals.squeeze(), self.norm["type"], self.norm["params"], self.clim
            ),
            "synth": WakeGANDataset.transform_back(
                synths.squeeze(), self.norm["type"], self.norm["params"], self.clim
            ),
        }
        self.metadatas_test = metadatas

        self.mse_test(self.images_test["real"], self.images_test["synth"])

        self.log(
            "rmse_test_epoch",
            self.mse_test,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            "fid_test_epoch",
            self.fid_test.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return {
            "reals": reals,
            "synths": synths,
            "metadatas": metadatas,
        }

    def predict_step(self, batch, batch_idx):
        inflows = batch
        return self(inflows)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=float(self.lr), betas=self.betas
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=float(self.lr), betas=self.betas
        )

        lr_scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_g, T_max=50, eta_min=1e-7, verbose=False
        )
        lr_scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_d, T_max=50, eta_min=1e-7, verbose=False
        )

        return [opt_d, opt_g], [lr_scheduler_d, lr_scheduler_g]

    def _train_discriminator(self, inflows, synths, reals) -> None:
        """Train Discriminator: max log(D(x)) + log(1 - D(G(z)))"""

        pred_real = self.discriminator(reals, inflows)
        pred_synth = self.discriminator(synths, inflows)

        loss_real = self.loss["bce"](pred_real, torch.ones_like(pred_real))
        loss_synth = self.loss["bce"](pred_synth, torch.zeros_like(pred_synth))

        d_loss = loss_real + loss_synth

        self.log("d_loss_real", loss_real, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "d_loss_synth", loss_synth, on_step=False, on_epoch=True, prog_bar=False
        )

        return d_loss

    def _train_generator(self, inflows, synths, reals) -> None:
        """Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))"""

        pred_synth = self.discriminator(synths, inflows)

        loss_adv = 1e-3 * self.loss["bce"](pred_synth, torch.ones_like(pred_synth))
        loss_mse = self.loss["mse"](reals, synths)

        g_loss = (1 - self.f_mse) * loss_adv + self.f_mse * loss_mse

        self.log(
            "g_loss_adv",
            (1 - self.f_mse) * loss_adv,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "g_loss_mse",
            self.f_mse * loss_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("g_loss", g_loss, on_step=False, on_epoch=True, prog_bar=False)

        return g_loss

    def _set_hparams(self, config, norm_params):
        self.channels: int = config["data"]["channels"]
        self.size: tuple = config["data"]["size"]
        self.data_dir: Dict = {
            "train": os.path.join("data", "preprocessed", "tracked", "train"),
            "test": os.path.join("data", "preprocessed", "tracked", "test"),
        }
        self.data_config: Dict = config["data"]

        if not norm_params:
            with open(os.path.join("data", "aux", "norm_params.json")) as file:
                norm_params = json.load(file)

        self.norm = {
            "type": config["data"]["normalization"]["type"],
            "params": norm_params,
        }
        self.clim = [config["data"]["figures"]["clim_ux"]]

        self.device_name: str = config["train"]["device"]
        self.lr: float = config["train"]["lr"]
        self.minibatch_size: int = config["train"]["batch_size"]
        self.f_mse: int = config["train"]["f_mse"]
        self.betas: tuple = (0.5, 0.999)
        self.workers: int = config["train"]["num_workers"]

        self.feat_gen = config["models"]["f_g"]
        self.feat_disc = config["models"]["f_d"]

        self.wt_d = config["data"]["wt_diam"]
        self.limits = config["data"]["lim_around_wt"]
