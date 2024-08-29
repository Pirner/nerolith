import cv2
import numpy as np


class VisionRendering:
    """
    Central class for vision rendering of any processes within the website
    """
    def __init__(self, alpha=1.0, beta=0.5):
        """
        init the class
        :param alpha: when putting add weighted on it, first image intensity
        :param beta: when putting add weighted on it, second image intensity
        """
        self.alpha = alpha
        self.beta = beta

    def render_binary_mask(self, im: np.ndarray, mask: np.ndarray):
        """
        render a binary mask onto the image, make sure the sizes match
        :param im: image to render on
        :param mask: mask to render on
        :return:
        """
        assert (im.shape[0], im.shape[1]) == (mask.shape[0], mask.shape[1])

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask[:, :, 0] = 0
        mask[:, :, 1] = 0
        rendered = cv2.addWeighted(im, self.alpha, mask, self.beta, 0)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < 30 or w < 30:
                continue
            rendered = cv2.rectangle(rendered, (x, y), (x + w, y + h), (0, 0, 255), 5)

        return rendered
