"""Module with WakeGAN class that defines general methods for training the network"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "12/22"

import logging
import os
from typing import List, Dict

from torch import Tensor
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics

from src.data.dataset import WakeGANDataset

from src.models import dcgan
from src.visualization import plots
from src.utils import utils


class WakeGAN:
    def __init__(self, config: Dict, logger: logging.Logger):

        self.logger = logger
        self.channels: int = config["data"]["channels"]
        self.size: tuple = config["data"]["size"]
        self.data_dir: Dict = {
            "train": os.path.join("data", "preprocessed", "tracked", "train"),
            "test": os.path.join("data", "preprocessed", "tracked", "test"),
        }
        self.data_config: Dict = config["data"]

        self.device_name: str = config["train"]["device"]
        self.lr: float = config["train"]["lr"]
        self.minibatch_size: int = config["train"]["batch_size"]

        self.f_mse: int = config["train"]["f_mse"]
        self.betas: tuple = (0.5, 0.999)
        self.workers: int = config["train"]["num_workers"]

        self.load = config["models"]["load"]
        self.save = config["models"]["save"]
        self.save_every = config["models"]["save_every"]

        self.net_name = {}
        self.net_name["generator"] = config["models"]["name_gen"]
        self.net_name["discriminator"] = config["models"]["name_disc"]

        self.feat_gen = config["models"]["f_g"]
        self.feat_disc = config["models"]["f_d"]

        self.rmse = {"train": [], "dev": []}

        torch.manual_seed(4)
        torch.use_deterministic_algorithms(True)

    def set_device(self) -> None:

        if not torch.cuda.is_available():
            self.device = "cpu"
            device_name = "CPU"
        else:
            if torch.cuda.device_count() == 1:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device(self.device_name)
            device_name = torch.cuda.get_device_name()

        self.logger.info(
            f"Using device: {device_name}"
            f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB\n"
        )

    def preprocess_dataset(self) -> None:
        self.dataset = {}
        self.dataset["train"] = WakeGANDataset(
            data_dir=self.data_dir["train"],
            config=self.data_config,
            dataset_type="train",
            save_norm_params=True if self.save else False,
        )
        self.dataset["dev"] = WakeGANDataset(
            data_dir=self.data_dir["test"],
            config=self.data_config,
            dataset_type="dev",
            norm_params=self.dataset["train"].norm_params,
        )

        plots.plot_histogram(self.dataset["train"])
        plots.plot_histogram(self.dataset["dev"])

    def initialize_models(self) -> None:
        self.generator = dcgan.Generator(
            self.channels,
            self.size[0],
            self.feat_gen,
        ).to(self.device)

        self.discriminator = dcgan.Discriminator(
            self.channels,
            self.feat_disc,
            self.size[0],
        ).to(self.device)

        self.logger.info(
            f"Initializing Generator with {sum(p.numel() for p in self.generator.parameters())} params\n"
            f"Initializing Discriminator with {sum(p.numel() for p in self.discriminator.parameters())} params\n"
        )

    def define_loss_and_optimizer(self) -> None:
        self.criterion = torch.nn.BCELoss()
        self.mse_train = torch.nn.MSELoss()

        self.optimizer = {
            "generator": torch.optim.Adam(
                self.generator.parameters(), lr=float(self.lr), betas=self.betas
            ),
            "discriminator": torch.optim.Adam(
                self.discriminator.parameters(), lr=float(self.lr), betas=self.betas
            ),
        }

        self.scheduler = {
            "generator": torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer["generator"], T_max=50, eta_min=1e-7, verbose=True
            ),
            "discriminator": torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer["discriminator"], T_max=50, eta_min=1e-7, verbose=True
            ),
        }

        self.logger.info(
            f"Using {self.criterion}\n"
            f"Using Adam optimizer for generator with lr: {self.lr} and betas {self.betas}\n"
            f"Using Adam optimizer for discriminator with lr: {self.lr} and betas: {self.betas}\n"
        )

    def load_pretrained_models(self) -> None:
        self._load_model(
            self.generator, self.net_name["generator"], self.optimizer["generator"]
        )
        self._load_model(
            self.discriminator,
            self.net_name["discriminator"],
            self.optimizer["discriminator"],
        )

    def train(self):
        self._set_dataset_loaders()
        n_minibatches = len(self.dataset["train"].loader)
        self.logger.info(
            f"Training set samples: {len(self.dataset['train'])}\n"
            f"Testing set samples: {len(self.dataset['dev'])}\n"
            f"Number of epochs: {self.epochs}\n"
            f"Mini batch size: {self.minibatch_size}\n"
            f"Number of training batches: {n_minibatches}\n"
        )

        plotters = {
            "metrics": plots.MetricsPlotter(self.epochs, self.dataset["train"].clim),
            "flow": plots.FlowImagePlotter(self.channels, self.dataset["train"].clim),
        }

        self._train_loop(plotters, n_minibatches)
        self._save_models()

    def _set_dataset_loaders(self) -> None:
        self.dataset["train"].set_loader(
            batch_size=self.minibatch_size, num_workers=self.workers
        )
        self.dataset["dev"].set_loader(
            batch_size=self.minibatch_size, num_workers=self.workers
        )
        return None

    def _train_loop(self, plotters: Dict, n_minibatches: int) -> None:
        self.generator.train()
        self.discriminator.train()
        for epoch in range(self.epochs):
            self.generator.running_loss_adv = 0
            self.generator.running_loss_mse = 0
            self.discriminator.running_loss = 0
            running_mse = 0
            for c, (images, inflows, _) in enumerate(self.dataset["train"].loader):
                images = images.float().to(self.device)
                inflows = inflows.float().to(self.device)
                synths = self.generator(inflows)

                self._train_discriminator(inflows, images, synths)
                self._train_generator(images, synths, inflows, epoch)

                running_mse += self._calculate_batch_mse(
                    images, synths, self.dataset["train"]
                )

            self.scheduler["discriminator"].step()
            self.scheduler["generator"].step()

            self.discriminator.loss.append(
                self.discriminator.running_loss / n_minibatches
            )
            self.generator.loss_adv.append(
                self.generator.running_loss_adv / n_minibatches
            )
            self.generator.loss_mse.append(
                self.generator.running_loss_mse / n_minibatches
            )

            self.rmse["train"].append(torch.sqrt(running_mse / n_minibatches).cpu())

            images_dev, synths_dev, rmse_dev, _ = self.evaluate_generator(
                self.dataset["dev"]
            )

            self.rmse["dev"].append(rmse_dev)

            plotters["metrics"].plot(
                {
                    "gen_adv": self.generator.loss_adv,
                    "gen_mse": self.generator.loss_mse,
                    "disc": self.discriminator.loss,
                },
                self.rmse,
                epoch,
            )

            plotters["flow"].plot(
                [
                    self.transform_back(
                        images[0][0].detach().cpu(), self.dataset["train"]
                    ),
                    self.transform_back(
                        synths[0][0].detach().cpu(), self.dataset["train"]
                    ),
                    self.transform_back(
                        images_dev[0][0].detach().cpu(), self.dataset["dev"]
                    ),
                    self.transform_back(
                        synths_dev[0][0].detach().cpu(), self.dataset["dev"]
                    ),
                ]
            )

            self.logger.info(
                f"{epoch + 1:03d}/{self.epochs}, "
                f"loss disc: {self.discriminator.loss[-1]:.2f} / "
                f"loss gen: {self.generator.loss_adv[-1]+self.generator.loss_mse[-1]:.2f}, "
                f"rmse train: {self.rmse['train'][-1]:.2f} / rmse dev: {self.rmse['dev'][-1]:.2f} "
            )

            if (epoch + 1) % self.save_every == 0:
                self._save_models()

        return None

    def _train_discriminator(self, inflows, images, synths) -> None:
        """Train Discriminator: max log(D(x)) + log(1 - D(G(z)))"""

        pred_real = self.discriminator(images, inflows)
        pred_synth = self.discriminator(synths, inflows)

        loss_real = self.criterion(pred_real, torch.ones_like(pred_real))
        loss_synth = self.criterion(pred_synth, torch.zeros_like(pred_synth))

        loss_d = loss_real + loss_synth

        self.optimizer["discriminator"].zero_grad()
        loss_d.backward(retain_graph=True)
        self.optimizer["discriminator"].step()

        self.discriminator.running_loss += loss_d.item()

        return None

    def _train_generator(self, images, synths, inflows, epoch) -> None:
        """Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))"""

        pred_synth = self.discriminator(synths, inflows)

        loss_adv = self.criterion(pred_synth, torch.ones_like(pred_synth))
        loss_mse = self.mse_train(images, synths)

        loss = (1 - self.f_mse) * loss_adv + self.f_mse * loss_mse
        self.optimizer["generator"].zero_grad()
        loss.backward()
        self.optimizer["generator"].step()

        self.generator.running_loss_adv += loss_adv.item()
        self.generator.running_loss_mse += loss_mse.item()

        return None

    def _save_models(self):
        self._save_model(
            self.generator,
            self.net_name["generator"],
            self.optimizer["generator"],
            self.logger,
        )
        self._save_model(
            self.discriminator,
            self.net_name["discriminator"],
            self.optimizer["discriminator"],
            self.logger,
        )

        return None

    @staticmethod
    def _save_model(model, name, optimizer, logger):
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join("models", name),
        )
        logger.info(f"Saved model {name} to disk"),

    @torch.no_grad()
    def evaluate_generator(self, dataset: WakeGANDataset) -> tuple((Tensor, float)):
        self.generator.eval()
        running_mse = 0
        for (images, inflows, metadatas) in dataset.loader:
            inflows = inflows.float().to(self.device)
            images = images.to(self.device)

            synths = self.generator(inflows)
            running_mse += self._calculate_batch_mse(images, synths, dataset)

        mse = running_mse / len(dataset.loader)
        rmse = torch.sqrt(mse).cpu()
        return images, synths, rmse, metadatas

    def _calculate_batch_mse(
        self, images: Tensor, synths: Tensor, dataset: WakeGANDataset
    ) -> Tensor:
        mse = 0
        for (img, synth) in zip(images.detach(), synths.detach()):
            img = self.transform_back(img, dataset)
            synth = self.transform_back(synth, dataset)

            mse += utils.calculate_mse(
                img.flatten(), synth.flatten(), np.prod(self.size)
            )
        mse /= len(images)
        return mse

    @staticmethod
    def transform_back(image: Tensor, dataset: WakeGANDataset) -> Tensor:
        image = WakeGANDataset.unnormalize_image(
            dataset.norm_type, dataset.norm_params, image
        )
        image = WakeGANDataset.rescale_back_to_velocity(image, dataset.clim)
        return image

    def plot_monitor_figures():
        ...

    def save_models():
        ...

    def _load_model(
        self, model: torch.nn.Module, name: str, optimizer: torch.optim.Optimizer
    ) -> None:
        checkpoint = torch.load(os.path.join("models", name), map_location=self.device)

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr

        self.logger.info(f"Loaded model {name} from disk")


class LitWakeGAN(pl.LightningModule):
    def __init__(self, config: Dict, norm_params: Dict):

        super().__init__()
        # self.save_hyperparameters()
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
            prog_bar=True,
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
            prog_bar=True,
        )

    def training_epoch_end(self, outputs):
        self.log("rmse_train_epoch", self.mse_train, prog_bar=True)

    def validation_epoch_end(self, outputs):
        self.log("rmse_dev_epoch", self.mse_dev, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        inflows, _ = batch
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

    def _train_discriminator(self, inflows, images) -> None:
        """Train Discriminator: max log(D(x)) + log(1 - D(G(z)))"""

        pred_real = self.discriminator(images, inflows)
        pred_synth = self.discriminator(self.synths, inflows)

        loss_real = self.loss["bce"](pred_real, torch.ones_like(pred_real))
        loss_synth = self.loss["bce"](pred_synth, torch.zeros_like(pred_synth))

        d_loss = loss_real + loss_synth

        self.log("d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True)

        return d_loss

    def _train_generator(self, images, inflows) -> None:
        """Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))"""

        pred_synth = self.discriminator(self.synths, inflows)

        loss_adv = self.loss["bce"](pred_synth, torch.ones_like(pred_synth))
        loss_mse = self.loss["mse"](images, self.synths)

        g_loss = (1 - self.f_mse) * loss_adv + self.f_mse * loss_mse

        self.log("g_loss", g_loss, on_step=False, on_epoch=True, prog_bar=True)

        return g_loss

    def _set_hparams(self, config, norm_params):
        # self.logger = logger
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
