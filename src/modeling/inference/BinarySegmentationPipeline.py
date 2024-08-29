import cv2
import numpy as np
import torch

from src.config.DTO import ModelConfig
from src.modeling.factory import ModelFactory
from src.modeling.transformation.BinarySegmentationTransformation import BinarySegmentationTransform


class BinarySegmentationInferencePipeline:
    """
    This class is for perform inference and manage all the stuff around it for binary segmentation of images.
    This pipeline per default loads the model and runs all the inference stuff on the cpu.
    Therefor a model configuration needs to be provided.
    """
    def __init__(self, config: ModelConfig, model_path: str, device='cpu', threshold=0.5):
        """
        base constructor
        :param config: model configuration to construct the model itself
        :param model_path: path to the model file - use a pth file to restore the weights
        :param device: computing device to run the predictions on
        :param threshold: threshold for creating the segmentation map after the prediction
        """
        self.config = config
        self.model_path = model_path

        self.model = ModelFactory.create_model(config=config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transforms = BinarySegmentationTransform.get_test_transform(im_h=config.im_height, im_w=config.im_width)
        self.threshold = threshold
        self.device = device

    def predict_image(self, im: np.ndarray):
        """
        predict an image and create the mask from it
        :param im: image to run inference on
        :return:
        """
        x = self.transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))['image']
        x = torch.unsqueeze(x, 0)
        y = self.model(x)
        y = y.cpu().detach().numpy()
        y = np.squeeze(y, axis=0)
        y = np.squeeze(y, axis=0)

        segmentation_map = (y >= self.threshold).astype(np.uint8) * 255
        return segmentation_map
