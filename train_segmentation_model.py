import pytorch_lightning as pl
import torch

from src.config.DTO import ModelConfig
from src.data.io import DataIO
from src.modeling.callbacks.CSVLogging import CSVLoggerCallback
from src.modeling.callbacks.ModelCheckpoint import SemSegModelCheckpointCallback
from src.modeling.datasets.BinarySegmentationDataset import BinarySegmentationDataset
from src.modeling.factory import ModelFactory
from src.modeling.training.BinarySegmentationTrainingModule import BinarySegmentationTrainer
from src.modeling.transformation.BinarySegmentationTransformation import BinarySegmentationTransform


def create_callbacks(experiment_path: str):
    """
    create the callbacks for training the model
    :param experiment_path: experiment path to store the data being produced in
    :return:
    """
    cbs = [
        SemSegModelCheckpointCallback(experiment_path=experiment_path),
        CSVLoggerCallback(experiment_path=experiment_path),
    ]
    return cbs


def main():
    dataset_path = r'C:\data\TACO\data'
    experiment_path = r'C:\project_data\nerolith\models'

    data = DataIO.read_dataset(dataset_path)

    n_train = int(len(data) * 0.8)
    train_data = data[:n_train]
    val_data = data[n_train:]

    config = ModelConfig(
        backbone='resnet18',
        architecture='fpn',
        n_classes=1,
        im_height=800,
        im_width=800,
    )
    train_transform = BinarySegmentationTransform.get_train_transforms(im_h=config.im_height, im_w=config.im_width)
    val_transform = BinarySegmentationTransform.get_val_transform(im_h=config.im_height, im_w=config.im_width)

    train_ds = BinarySegmentationDataset(data=train_data, config=config, transform=train_transform)
    val_ds = BinarySegmentationDataset(data=val_data, config=config, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=2,
        num_workers=1
    )

    model = ModelFactory.create_model(config=config)
    # create the pytorch_lightning module
    trainer = BinarySegmentationTrainer(model=model)
    pl_module_trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100,
        callbacks=create_callbacks(experiment_path),
        logger=False,
        enable_checkpointing=False,
    )

    pl_module_trainer.fit(trainer, train_loader, val_loader)


if __name__ == '__main__':
    main()
