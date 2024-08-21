import segmentation_models_pytorch as smp

from src.config.DTO import ModelConfig


class ModelFactory(object):
    @staticmethod
    def create_model(config: ModelConfig):
        """
        create model with factory pattern
        :param config:
        :return:
        """
        if config.architecture.lower() == 'fpn':
            model = ModelFactory.build_fpn(config)
        else:
            raise ValueError('no valid architecture for model provided.')

        return model

    @staticmethod
    def build_fpn(config: ModelConfig):
        """
        build a feature pyramid network segmentation model
        :param config:
        :return:
        """
        model = smp.FPN(config.backbone, classes=config.n_classes)
        return model
