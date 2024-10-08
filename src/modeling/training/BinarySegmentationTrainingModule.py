import pytorch_lightning as pl
import torch
from torchmetrics import JaccardIndex

from src.modeling.losses.DiceLoss import DiceLoss


class BinarySegmentationTrainer(pl.LightningModule):
    """
    this is a training module for training binary segmentation models
    It is important for training to know the number of classes, this is why this procedure has been laid out
    into two separate classes for readability.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.jaccard = JaccardIndex(task='binary')
        self.dice_loss = DiceLoss()

    def forward(self, sample):
        x = sample
        logit = self.model(x)
        return logit

    def training_step(self, batch, batch_idx):
        sample, y = batch
        predictions = self(sample)
        loss = self.dice_loss(predictions, y)
        jaccard_score = self.jaccard(predictions, y)

        self.log('train_loss', loss.item(), on_epoch=True, prog_bar=True)
        self.log('train_jaccard', jaccard_score.item(), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sample, y = batch
        predictions = self(sample)
        loss = self.dice_loss(predictions, y)
        jaccard_score = self.jaccard(predictions, y)

        self.log('val_loss', loss.item(), on_epoch=True, prog_bar=True)
        self.log('val_jaccard', jaccard_score.item(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        return optimizer
