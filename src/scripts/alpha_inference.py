import os
import json

import cv2
import numpy as np
import torch

from src.config.DTO import ModelConfig
from src.modeling.factory import ModelFactory
from src.modeling.transformation.BinarySegmentationTransformation import BinarySegmentationTransform
from src.vision.rendering import VisionRendering


def main():
    im_path = r'C:\data\TACO\data\batch_13\000042.jpg'
    experiment_path = r'C:\project_data\nerolith\alpha_model'
    with open(os.path.join(experiment_path, 'config.json')) as json_data:
        config = json.load(json_data)
        json_data.close()

    renderer = VisionRendering()
    config = ModelConfig(**config)
    model = ModelFactory.create_model(config=config)
    model.load_state_dict(torch.load(os.path.join(experiment_path, 'model.pth')))
    model.eval()
    im = cv2.imread(im_path)
    transforms = BinarySegmentationTransform.get_test_transform(im_h=config.im_height, im_w=config.im_width)

    x = transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))['image']
    x = torch.unsqueeze(x, 0)
    y = model(x)
    y = y.cpu().detach().numpy()
    y = np.squeeze(y, axis=0)
    y = np.squeeze(y, axis=0)

    segmentation_map = (y >= 0.5).astype(np.uint8) * 255
    segmentation_map = cv2.resize(segmentation_map, (im.shape[1], im.shape[0]), cv2.INTER_LINEAR_EXACT)
    overlay = renderer.render_binary_mask(im=im, mask=segmentation_map)

    cv2.imwrite(r'C:\project_data\nerolith\results\segmap.png', segmentation_map)
    cv2.imwrite(r'C:\project_data\nerolith\results\overlay.png', overlay)
    cv2.imwrite(r'C:\project_data\nerolith\results\src_im.png', im)

    exit(0)


if __name__ == '__main__':
    main()
