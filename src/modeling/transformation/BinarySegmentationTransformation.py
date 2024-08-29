import albumentations as A
from albumentations.pytorch import ToTensorV2


class BinarySegmentationTransform:
    @staticmethod
    def get_train_transforms(im_h: int, im_w: int):
        """
        get training data transforms
        :param im_h: image height
        :param im_w: image width
        :return:
        """
        t_train = A.Compose([
            A.Resize(width=im_w, height=im_h, always_apply=True),
            A.HorizontalFlip(p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
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
        t_val = A.Compose([
            A.Resize(width=im_w, height=im_h, always_apply=True),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        return t_val

    @staticmethod
    def get_test_transform(im_h: int, im_w: int):
        """
        get validation data transforms
        :param im_h: image height
        :param im_w: image width
        :return:
        """
        t_val = A.Compose([
            A.Resize(width=im_w, height=im_h, always_apply=True),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        return t_val
