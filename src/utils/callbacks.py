"""Module with Pytorch Lightning callbacks"""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "01/23"

from pytorch_lightning import callbacks


class CustomCallback(callbacks.Callback):
    def on_train_start(self, trainer, pl_module):
        pass

    def on_train_end(self, trainer, pl_module):
        pass
