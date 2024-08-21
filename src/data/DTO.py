from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np


@dataclass
class DataPoint:
    im_path: str
    annotation: List[Dict]
    image_metadata: Dict

    def create_binary_mask(self) -> np.ndarray:
        """
        create a binary mask with the annotations stored
        :return:
        """
        h = self.image_metadata['height']
        w = self.image_metadata['width']
        mask = np.zeros((h, w, 1))

        for ann in self.annotation:
            contours = []
            for cnt in ann['segmentation']:
                cnt_parsed = []
                assert (len(cnt) % 2) == 0

                chunks = (len(cnt) - 1) // 2 + 1
                for i in range(chunks):
                    batch = cnt[i * 2:(i + 1) * 2]
                    cnt_parsed.append(np.array((int(batch[0]), int(batch[1]))))
                cnt_parsed = np.array(cnt_parsed)
                contours.append(cnt_parsed)
            mask = cv2.drawContours(mask, contours, -1, 255, -1)
        return mask
