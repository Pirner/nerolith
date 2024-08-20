import os
import glob
import json
from typing import Dict, List
from pathlib import Path

from src.data.DTO import DataPoint


class DataIO:
    @staticmethod
    def transform_im_path_into_data_point(
            im_path: str,
            annotations: List[Dict],
            images_metadata: List[Dict],
    ) -> DataPoint:
        """
        transform image path into data point
        :param im_path: image path to transform
        :param annotations: annotations to extract from
        :param images_metadata: metadata from the images
        :return:
        """
        batch_id = im_path.split(os.sep)[-2]
        filename = im_path.split(os.sep)[-1]

        image_filename = os.path.join('{}/{}'.format(batch_id, filename))

        image_metadata = list(filter(lambda x: image_filename in x['file_name'], images_metadata))
        assert len(image_metadata) == 1
        image_metadata = image_metadata[0]
        image_id = image_metadata['id']

        image_annotations = list(filter(lambda x: x['image_id'] == image_id, annotations))
        data_point = DataPoint(im_path, image_annotations, image_metadata)
        return data_point

    @staticmethod
    def read_dataset(dataset_path: str) -> List[DataPoint]:
        """
        read the dataset from disk
        :param dataset_path: path to the dataset to load
        :return:
        """
        annotation_path = os.path.join(dataset_path, 'annotations.json')
        im_paths = glob.glob(os.path.join(dataset_path, '**/*.jpg'), recursive=True)
        im_paths += glob.glob(os.path.join(dataset_path, '**/*.png'), recursive=True)
        im_paths += glob.glob(os.path.join(dataset_path, '**/*.tiff'), recursive=True)
        with open(annotation_path) as f:
            annotation_data = json.load(f)
        # create an image data point per image path
        annotations = annotation_data['annotations']
        image_metadata = annotation_data['images']

        data = [DataIO.transform_im_path_into_data_point(x, annotations, image_metadata) for x in im_paths]
        return data
