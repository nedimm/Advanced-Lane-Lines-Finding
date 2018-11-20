import glob
import collections

import cv2
import matplotlib.image as mpimg


class ImageReader:
    def __init__(self, read_mode, regex='../camera_cal/calibration*.jpg'):
        self.regex = regex
        self.set_image_names()
        self.read_mode = read_mode

    def set_image_names(self):
        self.image_names = glob.glob(self.regex)

    def images(self):
        for file_name in self.image_names:
            image = collections.namedtuple('ImageFile', 'name image')
            if self.read_mode == 'RGB':
                # pylab_img[::-1,:,::-1] == cv_img
                yield image(file_name, mpimg.imread(file_name))
            elif self.read_mode == 'BGR':
                yield image(file_name, cv2.imread(file_name))
            else:
                raise ValueError('Only RGB or BGR read mode are supported.')

    def get_gray_image(self, image):
        if self.read_mode == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

