import os
import glob
import json

import cv2
import numpy as np
from tqdm import tqdm

from src.config.DTO import ModelConfig
from src.modeling.inference.BinarySegmentationPipeline import BinarySegmentationInferencePipeline
from src.vision.rendering import VisionRendering


def main():
    experiment_path = r'C:\project_data\nerolith\alpha_model'
    with open(os.path.join(experiment_path, 'config.json')) as json_data:
        config = json.load(json_data)
        json_data.close()

    renderer = VisionRendering()
    config = ModelConfig(**config)
    pipeline = BinarySegmentationInferencePipeline(config=config, model_path=os.path.join(experiment_path, 'model.pth'))

    src_directory = r'C:\data\TACO\data\batch_13'
    im_paths = glob.glob(os.path.join(src_directory, '**/*.jpg'), recursive=True)
    for i, im_path in tqdm(enumerate(im_paths), total=len(im_paths)):

        im = cv2.imread(im_path)
        y = pipeline.predict_image(im=im)

        segmentation_map = (y >= 0.5).astype(np.uint8) * 255
        segmentation_map = cv2.resize(segmentation_map, (im.shape[1], im.shape[0]), cv2.INTER_LINEAR_EXACT)
        overlay = renderer.render_binary_mask(im=im, mask=segmentation_map)

        cv2.imwrite(r'C:\project_data\nerolith\results\{:04d}_segmap.png'.format(i), segmentation_map)
        cv2.imwrite(r'C:\project_data\nerolith\results\{:04d}_overlay.png'.format(i), overlay)
        cv2.imwrite(r'C:\project_data\nerolith\results\{:04d}_src_im.png'.format(i), im)


if __name__ == '__main__':
    main()
