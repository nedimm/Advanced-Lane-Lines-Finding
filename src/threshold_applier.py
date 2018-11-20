import glob

import cv2
import matplotlib.image as mpimg
import numpy as np
from src.image_reader import ImageReader
from src.image_plotter import ImagePlotter

class ThresholdApplier:
    def __init__(self, threshold_gradient=(20, 100), threshold_magnitude=(20, 100),
                 threshold_direction=(0.7, 1.3), threshold_s=(170, 255)):
        self.threshold_gradient = threshold_gradient
        self.threshold_magnitude = threshold_magnitude
        self.threshold_direction = threshold_direction
        self.threshold_s = threshold_s


    def apply_stacked_thresholds(self, image):
        c_mask, g_mask, s = self.isolate_lane_line_pixels(image)
        return np.dstack((np.zeros_like(s), g_mask, c_mask))

    def apply_combined_thresholds(self, image):
        c_mask, g_mask, _ = self.isolate_lane_line_pixels(image)

        mask = np.zeros_like(g_mask)
        mask[(g_mask == 1) | (c_mask == 1)] = 1
        return mask


    def isolate_lane_line_pixels(self, image):
        hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
        s = hls[:, :, 2]
        grad_x = self._get_gradient_absolute_value_mask(s, 1, 0, threshold=self.threshold_gradient)
        grad_y = self._get_gradient_absolute_value_mask(s, 1, 1, threshold=self.threshold_gradient)
        magnitude = self._get_gradient_magnitude_mask(s, threshold=self.threshold_magnitude)
        direction = self._get_gradient_direction_mask(s, threshold=self.threshold_direction)
        g_mask = np.zeros_like(s)
        g_mask[((grad_x == 1) & (grad_y == 1)) | ((magnitude == 1) & (direction == 1))] = 1

        c_mask = self._get_mask(s, threshold=self.threshold_s)
        return c_mask, g_mask, s


    def _get_gradient_absolute_value_mask(self, image, dx, dy, threshold, sobel_ksize=3):
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=sobel_ksize))
        return self._get_mask(sobel, threshold)

    def _get_mask(self, image, threshold):
        scaled_image = np.uint8(255 * image / np.max(image))
        mask = np.zeros_like(scaled_image)
        mask[(scaled_image >= threshold[0]) & (scaled_image <= threshold[1])] = 1
        return mask

    def _get_gradient_magnitude_mask(self, image, threshold, sobel_ksize=3):
        x, y = self._get_gradients(image, sobel_ksize)
        magnitude = np.sqrt(x ** 2 + y ** 2)
        return self._get_mask(magnitude, threshold)

    def _get_gradient_direction_mask(self, image, threshold, sobel_ksize=3):
        x, y = self._get_gradients(image, sobel_ksize)
        direction_mask = np.arctan2(np.absolute(y), np.absolute(x))
        return self._get_mask(direction_mask, threshold)

    def _get_gradients(self, image, sobel_ksize):
        x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        return x, y


if __name__ == "__main__":
    ir = ImageReader(read_mode='RGB', regex='../test_images/test*.jpg')
    ip = ImagePlotter()

    le = ThresholdApplier()
    images_to_show = []
    for name, image in ir.images():
        ip.add_to_plot(image, name, 'out', False)

        stacked = le.apply_stacked_thresholds(image)
        ip.add_to_plot(stacked, name + " stacked", 'out', True)

        combined = le.apply_combined_thresholds(image)
        ip.add_to_plot(combined, name + " combined", 'out', True)

    ip.plot('out', 3)
