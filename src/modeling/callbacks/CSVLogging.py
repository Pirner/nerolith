import os

import pandas as pd
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl


class CSVLoggerCallback(Callback):
    """
    provide csv logging for a current experiment, just everything
    """
    def __init__(self, experiment_path: str):
        """
        create the callback
        :param experiment_path: where to save the logs
        """
        self.experiment_path = experiment_path
        self.metrics = {}

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        write the logs on validation epoch end
        :param trainer: trainer which is running the experiment
        :param pl_module: pytorch lightning module which holds the logic
        :return:
        """
        if trainer.sanity_checking:
            return

        for key in trainer.callback_metrics:
            if key not in self.metrics:
                self.metrics[key] = []
            val = trainer.callback_metrics[key].cpu().detach().numpy()
            self.metrics[key].append(val)

        df = pd.DataFrame(self.metrics)
        df.to_csv(os.path.join(self.experiment_path, 'training_logs.csv'), sep=';', index=False)