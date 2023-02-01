"""Module with WakeGAN class that defines general methods for training the network"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

from typing import Dict
import os

import pytorch_lightning as pl
import torch
import torchmetrics

from src.data.dataset import WakeGANDataset
from src.models import dcgan


class WakeGAN(pl.LightningModule):
    def __init__(self, config: Dict, norm_params: Dict):

        super().__init__()
        self._set_hparams(config, norm_params)

        torch.manual_seed(4)
        torch.use_deterministic_algorithms(True)

        self.generator = dcgan.Generator(self.channels, self.size[0], self.feat_gen)
        self.discriminator = dcgan.Discriminator(
            self.channels, self.feat_disc, self.size[0]
        )

        self.loss = {
            "bce": torch.nn.BCELoss(),
            "mse": torch.nn.MSELoss(),
        }
        self.mse_train = torchmetrics.MeanSquaredError(squared=False)
        self.mse_dev = torchmetrics.MeanSquaredError(squared=False)
        self.mse_test = torchmetrics.MeanSquaredError(squared=False)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):

        self.reals, inflows, _ = batch

        self.synths = self(inflows)

        if optimizer_idx == 0:
            loss = self._train_discriminator(inflows, self.reals)

        if optimizer_idx == 1:
            loss = self._train_generator(self.reals, inflows)

        self.images = {
            "real": WakeGANDataset.transform_back(
                self.reals.clone().squeeze(),
                self.norm["type"],
                self.norm["params"],
                self.clim,
            ),
            "synth": WakeGANDataset.transform_back(
                self.synths.clone().squeeze(),
                self.norm["type"],
                self.norm["params"],
                self.clim,
            ),
        }

        self.mse_train(
            self.images["real"],
            self.images["synth"],
        )

        self.log(
            "mse_train_step",
            self.mse_train,
            prog_bar=False,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        reals, inflows, _ = batch
        synths = self(inflows)

        self.images_dev = {
            "real": WakeGANDataset.transform_back(
                reals.squeeze(), self.norm["type"], self.norm["params"], self.clim
            ),
            "synth": WakeGANDataset.transform_back(
                synths.squeeze(), self.norm["type"], self.norm["params"], self.clim
            ),
        }

        self.mse_dev(
            self.images_dev["real"],
            self.images_dev["synth"],
        )

        self.log(
            "mse_dev_step",
            self.mse_dev,
            prog_bar=False,
        )

    def test_step(self, batch, batch_idx):
        reals, inflows, metadatas = batch
        synths = self(inflows)

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
        self.log("rmse_test_step", self.mse_test, prog_bar=False)

    def predict_step(self, batch, batch_idx):
        inflows, _ = batch
        return self(inflows)

    def training_epoch_end(self, outputs):
        self.log("rmse_train_epoch", self.mse_train, prog_bar=False)

    def validation_epoch_end(self, outputs):
        self.log("rmse_dev_epoch", self.mse_dev, prog_bar=True)

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

    def _train_discriminator(self, inflows, images) -> None:
        """Train Discriminator: max log(D(x)) + log(1 - D(G(z)))"""

        pred_real = self.discriminator(images, inflows)
        pred_synth = self.discriminator(self.synths, inflows)

        loss_real = self.loss["bce"](pred_real, torch.ones_like(pred_real))
        loss_synth = self.loss["bce"](pred_synth, torch.zeros_like(pred_synth))

        d_loss = loss_real + loss_synth

        self.log("d_loss_real", loss_real, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "d_loss_synth", loss_synth, on_step=False, on_epoch=True, prog_bar=False
        )

        return d_loss

    def _train_generator(self, images, inflows) -> None:
        """Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))"""

        pred_synth = self.discriminator(self.synths, inflows)

        loss_adv = 1e-3 * self.loss["bce"](pred_synth, torch.ones_like(pred_synth))
        loss_mse = self.loss["mse"](images, self.synths)

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
        self.log("g_loss", g_loss, on_step=False, on_epoch=True, prog_bar=True)

        return g_loss

    def _set_hparams(self, config, norm_params):
        self.channels: int = config["data"]["channels"]
        self.size: tuple = config["data"]["size"]
        self.data_dir: Dict = {
            "train": os.path.join("data", "preprocessed", "tracked", "train"),
            "test": os.path.join("data", "preprocessed", "tracked", "test"),
        }
        self.data_config: Dict = config["data"]
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
