import os

from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import torch


class SemSegModelCheckpointCallback(Callback):
    """
    this callback takes care of storing the model for segmentation in a convenient way.
    It also takes care to have the right logic of why and when models are stored.
    """
    def __init__(self, experiment_path: str):
        """

        :param experiment_path:
        """
        self.experiment_path = experiment_path

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        on validation end the performance of a model is taken and by demand the model is stored.
        :param trainer:
        :param pl_module:
        :return:
        """

        torch.save(pl_module.model.state_dict(), os.path.join(self.experiment_path, 'model.pth'))
