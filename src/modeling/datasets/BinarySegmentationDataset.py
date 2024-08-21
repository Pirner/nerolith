from typing import List

from PIL import Image
from torch.utils.data import Dataset

from src.config.DTO import ModelConfig
from src.data.DTO import DataPoint
from src.vision.io import VisionIO


class BinarySegmentationDataset(Dataset):
    def __init__(self, data: List[DataPoint], config: ModelConfig, transform=None):
        """
        create a segmentation dataset for binary segmentation
        :param data: data to consume
        :param config: configuration to derive the model from
        :param transform: transformation pipeline for the dataset
        """
        self.config = config
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        loads a data point in the lazy loading style
        :param item: item id to load
        :return:
        """
        data_point = self.data[item]

        im = VisionIO.read_image(im_path=data_point.im_path)
        mask = data_point.create_binary_mask()

        if self.transform is not None:
            aug = self.transform(image=im, mask=mask)
            im_out = aug['image']
            mask_out = aug['mask']

        else:
            im_out = Image.fromarray(im)
            mask_out = mask

        return im_out, mask_out
