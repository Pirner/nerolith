import cv2


class VisionIO:
    @staticmethod
    def read_image(im_path: str):
        """
        read an image from disk
        :param im_path: image path to read from
        :return:
        """
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im
