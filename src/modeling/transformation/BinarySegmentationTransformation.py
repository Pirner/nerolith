import torch
from torchvision.transforms import v2


class BinarySegmentationTransform:
    @staticmethod
    def get_train_transforms(im_h: int, im_w: int):
        """
        get training data transforms
        :param im_h: image height
        :param im_w: image width
        :return:
        """
        t_train = v2.Compose([
            v2.Resize(size=(im_h, im_w)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return t_train

    @staticmethod
    def get_val_transform(im_h: int, im_w: int):
        """
        get validation data transforms
        :param im_h: image height
        :param im_w: image width
        :return:
        """
        t_val = v2.Compose([
            v2.Resize(size=(im_h, im_w)),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return t_val
