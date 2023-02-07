"""Module with Pytorch Lightning callbacks"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "01/23"

import json
import os

from matplotlib import pyplot as plt
from neptune.new.types import File
from pytorch_lightning import callbacks
from torchvision import utils
import torch
import yaml

from src.visualization import plots

seed = 3

with open("config.yaml") as file:
    config = yaml.safe_load(file)


class LoggingCallback(callbacks.Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_train_start(self, trainer, pl_module):
        self.logger.info(f"Training {pl_module.__class__.__name__}:\n")

        if torch.cuda.is_available():
            self.logger.info(
                f"Using device: {torch.cuda.get_device_name()}"
                f" with {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.0f} GiB\n"
            )
        else:
            self.logger.info("Using device: CPU\n")

        self.logger.info(
            f"Initialized Generator with {sum(p.numel() for p in pl_module.generator.parameters())} params\n"
            f"Initialized Discriminator with {sum(p.numel() for p in pl_module.discriminator.parameters())} params\n"
        )

        self.logger.info(
            f"{pl_module.loss['bce']}\n"
            f"Adam optimizer for generator with initial lr: {pl_module.lr} and betas {pl_module.betas}\n"
            f"Adam optimizer for discriminator with initial lr: {pl_module.lr} and betas: {pl_module.betas}\n"
        )

        self.logger.info(
            f"Training set samples: {len(trainer.datamodule.dataset_train)}\n"
            f"Validation set samples: {len(trainer.datamodule.dataset_val)}\n"
            f"Testing set samples: {len(trainer.datamodule.dataset_test)}\n"
            f"Number of epochs: {trainer.max_epochs}\n"
            f"Mini batch size: {pl_module.minibatch_size}\n"
            f"Number of training batches: {len(trainer.train_dataloader)}\n"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        self.logger.info(
            f"{trainer.current_epoch + 1:03d}/{trainer.max_epochs}, "
            f"loss disc: {trainer.logged_metrics['d_loss_synth']+trainer.logged_metrics['d_loss_real']:.2f} / "
            f"loss gen: {trainer.logged_metrics['g_loss']:.2f}, "
            f"rmse train: {trainer.logged_metrics['rmse_train_epoch']:.2f} / rmse val: {trainer.logged_metrics['rmse_val_epoch']:.2f} "
        )


class PlottingCallback(callbacks.Callback):
    def __init__(self, enable_logger=False):
        super().__init__()
        self.metrics = {
            "loss_gen_adv": [],
            "loss_disc_real": [],
            "loss_disc_synth": [],
            "loss_gen_mse": [],
            "rmse_train": [],
            "rmse_val": [],
            "fid_train": [],
            "fid_val": [],
        }
        self.enable_logger = enable_logger

    def on_fit_start(self, trainer, pl_module):
        self.plotters = {
            "metrics": plots.MetricsPlotter(trainer.max_epochs, pl_module.clim),
            "flow": plots.FlowImagePlotter(pl_module.channels, pl_module.clim),
        }

    def on_train_epoch_end(self, trainer, pl_module):
        self.metrics["loss_gen_adv"].append(trainer.logged_metrics["g_loss_adv"].item())
        self.metrics["loss_disc_real"].append(
            trainer.logged_metrics["d_loss_real"].item()
        )
        self.metrics["loss_disc_synth"].append(
            trainer.logged_metrics["d_loss_synth"].item()
        )
        self.metrics["loss_gen_mse"].append(trainer.logged_metrics["g_loss_mse"].item())
        self.metrics["rmse_train"].append(
            trainer.logged_metrics["rmse_train_epoch"].item()
        )
        self.metrics["rmse_val"].append(trainer.logged_metrics["rmse_val_epoch"].item())
        self.metrics["fid_train"].append(
            trainer.logged_metrics["fid_train_epoch"].item()
        )
        self.metrics["fid_val"].append(trainer.logged_metrics["fid_val_epoch"].item())

        self.plotters["flow"].plot(
            [
                pl_module.images_train["real"][0].detach().cpu(),
                pl_module.images_train["synth"][0].detach().cpu(),
                pl_module.images_val["real"][1].detach().cpu(),
                pl_module.images_val["synth"][1].detach().cpu(),
            ]
        )
        fig_metrics, fig_losses = self.plotters["metrics"].plot(
            {
                "gen_adv": self.metrics["loss_gen_adv"],
                "gen_mse": self.metrics["loss_gen_mse"],
                "disc_synth": self.metrics["loss_disc_synth"],
                "disc_real": self.metrics["loss_disc_real"],
            },
            {
                "rmse": {
                    "train": self.metrics["rmse_train"],
                    "val": self.metrics["rmse_val"],
                },
                "fid": {
                    "train": self.metrics["fid_train"],
                    "val": self.metrics["fid_val"],
                },
            },
            trainer.current_epoch,
        )

        if trainer.current_epoch == trainer.max_epochs - 1 and self.enable_logger:
            pl_module.loggers[1].experiment["train"].log(File.as_image(fig_metrics))
            pl_module.loggers[1].experiment["train"].log(File.as_image(fig_losses))

        plt.close()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.state.fn != "fit":
            path = os.path.join("data", "generated", "validation")
            for real, synth, prec, angle, pos_x, pos_y, timestep in zip(
                outputs["reals"],
                outputs["synths"],
                outputs["metadatas"]["prec"],
                outputs["metadatas"]["angle"],
                outputs["metadatas"]["pos"][0],
                outputs["metadatas"]["pos"][1],
                outputs["metadatas"]["timestep"],
            ):
                if config["data"]["t_window"] == 1000:
                    filename = f"{prec.item()}_{angle}_({pos_x.item()},{pos_y.item()})_{timestep}.pt"
                else:
                    filename = (
                        f"{prec.item()}_{angle}_({pos_x.item()},{pos_y.item()}).pt"
                    )
                # utils.save_image(
                #     real, f"{os.path.join(path, 'real', filename)}", normalize=True
                # )
                # utils.save_image(
                #     synth, f"{os.path.join(path, 'synth', filename)}", normalize=True
                # )
                torch.save(real, f"{os.path.join(path, 'real', filename)}")
                torch.save(synth, f"{os.path.join(path, 'synth', filename)}")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.state.fn != "fit":
            path = os.path.join("data", "generated", "testing")
            for real, synth, prec, angle, pos_x, pos_y, timestep in zip(
                outputs["reals"],
                outputs["synths"],
                outputs["metadatas"]["prec"],
                outputs["metadatas"]["angle"],
                outputs["metadatas"]["pos"][0],
                outputs["metadatas"]["pos"][1],
                outputs["metadatas"]["timestep"],
            ):
                if config["data"]["t_window"] == 1000:
                    filename = f"{prec.item()}_{angle}_({pos_x.item()},{pos_y.item()})_{timestep}.pt"
                else:
                    filename = (
                        f"{prec.item()}_{angle}_({pos_x.item()},{pos_y.item()}).pt"
                    )
                # utils.save_image(
                #     real, f"{os.path.join(path, 'real', filename)}", normalize=True
                # )
                # utils.save_image(
                #     synth, f"{os.path.join(path, 'synth', filename)}", normalize=True
                # )
                torch.save(real, f"{os.path.join(path, 'real', filename)}")
                torch.save(synth, f"{os.path.join(path, 'synth', filename)}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.state.fn != "fit":
            mtdts = []
            for c, (prec, angle, pos_x, pos_y, timestep) in enumerate(
                zip(
                    pl_module.metadatas_val["prec"],
                    pl_module.metadatas_val["angle"],
                    pl_module.metadatas_val["pos"][0],
                    pl_module.metadatas_val["pos"][1],
                    pl_module.metadatas_val["timestep"],
                )
            ):

                mtdts.append(
                    {
                        "prec": prec.item(),
                        "angle": angle,
                        "pos": (pos_x.item(), pos_y.item()),
                        "timestep": timestep.item(),
                    }
                )

            torch.manual_seed(seed)
            indices = torch.randperm(len(pl_module.images_val["real"]))
            self.plotters_val = {
                "flow": plots.FlowImagePlotter(
                    pl_module.channels,
                    pl_module.clim,
                    monitor=False,
                    rmse=trainer.logged_metrics["rmse_val_epoch"].item(),
                    dataset="validation",
                ),
                "profiles": plots.ProfilesPlotter(
                    wt_d=pl_module.wt_d,
                    limits=pl_module.limits,
                    size=pl_module.size,
                    metadata=[mtdts[i] for i in indices[:4]],
                    dataset="validation",
                ),
            }

            images_to_plot = []
            for i in indices[:4]:
                images_to_plot.append(pl_module.images_val["real"][i].detach().cpu())
                images_to_plot.append(pl_module.images_val["synth"][i].detach().cpu())

            fig_flow = self.plotters_val["flow"].plot(images_to_plot)
            fig_profiles = self.plotters_val["profiles"].plot(images_to_plot)

            if self.enable_logger:
                pl_module.loggers[1].experiment["validation"].log(
                    File.as_image(fig_flow["img"])
                )
                pl_module.loggers[1].experiment["validation"].log(
                    File.as_image(fig_flow["err"])
                )
                pl_module.loggers[1].experiment["validation"].log(
                    File.as_image(fig_profiles)
                )

            with open(os.path.join("figures", "validation", "metrics.json"), "w") as f:
                json.dump(
                    {
                        "fid": trainer.logged_metrics["fid_val_epoch"].item(),
                        "rmse": trainer.logged_metrics["rmse_val_epoch"].item(),
                    },
                    f,
                )

            plt.close()

    def on_test_epoch_end(self, trainer, pl_module):

        mtdts = []
        for c, (prec, angle, pos_x, pos_y, timestep) in enumerate(
            zip(
                pl_module.metadatas_test["prec"],
                pl_module.metadatas_test["angle"],
                pl_module.metadatas_test["pos"][0],
                pl_module.metadatas_test["pos"][1],
                pl_module.metadatas_test["timestep"],
            )
        ):
            mtdts.append(
                {
                    "prec": prec.item(),
                    "angle": angle,
                    "pos": (pos_x.item(), pos_y.item()),
                    "timestep": timestep.item(),
                }
            )

        torch.manual_seed(seed)
        indices = torch.randperm(len(pl_module.images_test["real"]))
        self.plotters = {
            "flow": plots.FlowImagePlotter(
                pl_module.channels,
                pl_module.clim,
                monitor=False,
                rmse=trainer.logged_metrics["rmse_test_epoch"].item(),
                dataset="testing",
            ),
            "profiles": plots.ProfilesPlotter(
                wt_d=pl_module.wt_d,
                limits=pl_module.limits,
                size=pl_module.size,
                metadata=[mtdts[i] for i in indices[:4]],
                dataset="testing",
            ),
        }

        images_to_plot = []
        for i in indices[:4]:
            images_to_plot.append(pl_module.images_test["real"][i].detach().cpu())
            images_to_plot.append(pl_module.images_test["synth"][i].detach().cpu())

        self.plotters["flow"].plot(images_to_plot)
        self.plotters["profiles"].plot(images_to_plot)

        with open(os.path.join("figures", "testing", "metrics.json"), "w") as f:
            json.dump(
                {
                    "fid": trainer.logged_metrics["fid_test_epoch"].item(),
                    "rmse": trainer.logged_metrics["rmse_test_epoch"].item(),
                },
                f,
            )

        plt.close()
