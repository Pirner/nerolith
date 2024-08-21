from src.config.DTO import ModelConfig
from src.data.io import DataIO
from src.modeling.datasets.BinarySegmentationDataset import BinarySegmentationDataset
from src.modeling.transformation.BinarySegmentationTransformation import BinarySegmentationTransform


def main():
    dataset_path = r'C:\data\TACO\data'

    data = DataIO.read_dataset(dataset_path)

    n_train = int(len(data) * 0.8)
    train_data = data[:n_train]
    val_data = data[n_train:]

    config = ModelConfig(
        backbone='resnet18',
        architecture='fpn',
        n_classes=1,
        im_height=416,
        im_width=416,
    )
    train_transform = BinarySegmentationTransform.get_train_transforms(im_h=config.im_height, im_w=config.im_width)
    val_transform = BinarySegmentationTransform.get_val_transform(im_h=config.im_height, im_w=config.im_width)

    train_ds = BinarySegmentationDataset(data=train_data, config=config, transform=train_transform)
    val_ds = BinarySegmentationDataset(data=val_data, config=config, transform=val_transform)

    for x, y in train_ds:
        exit(0)


if __name__ == '__main__':
    main()
