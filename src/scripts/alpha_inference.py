import os
import json

import cv2
import numpy as np

from src.config.DTO import ModelConfig
from src.modeling.inference.BinarySegmentationPipeline import BinarySegmentationInferencePipeline
from src.vision.rendering import VisionRendering


def main():
    im_path = r'C:\data\TACO\data\batch_13\000042.jpg'
    experiment_path = r'C:\project_data\nerolith\alpha_model'
    with open(os.path.join(experiment_path, 'config.json')) as json_data:
        config = json.load(json_data)
        json_data.close()

    renderer = VisionRendering()
    config = ModelConfig(**config)
    pipeline = BinarySegmentationInferencePipeline(config=config, model_path=os.path.join(experiment_path, 'model.pth'))

    im = cv2.imread(im_path)
    y = pipeline.predict_image(im=im)

    segmentation_map = (y >= 0.5).astype(np.uint8) * 255
    segmentation_map = cv2.resize(segmentation_map, (im.shape[1], im.shape[0]), cv2.INTER_LINEAR_EXACT)
    overlay = renderer.render_binary_mask(im=im, mask=segmentation_map)

    cv2.imwrite(r'C:\project_data\nerolith\results\segmap.png', segmentation_map)
    cv2.imwrite(r'C:\project_data\nerolith\results\overlay.png', overlay)
    cv2.imwrite(r'C:\project_data\nerolith\results\src_im.png', im)


if __name__ == '__main__':
    main()
