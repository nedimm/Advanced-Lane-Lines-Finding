from collections import deque
import numpy as np


class LaneLine(object):
    """Implementation of a Lane Line object"""

    def __init__(self, x, y, heigth, width):
        self.heigth = heigth
        self.width = width
        self.coef = deque(maxlen=5)
        self.fit_points(x, y)
        self._meter_per_x_axis = 700
        self._meter_per_y_axis = 720
        # lane width assumed is 3.7 meters
        self._land_width = 3.7
        # assuming the lane is 30 m long
        self._image_lane_length = 30

    def fit_points(self, x, y):
        points = len(y) > 0 and (np.max(y) - np.min(y)) > self.heigth * 0.625
        no_coef = len(self.coef) == 0

        if points or no_coef:
            self.coef.append(np.polyfit(y, x, 2))

    def generate_points(self):
        y = np.linspace(0, self.heigth - 1, self.heigth)
        fit = np.array(self.coef).mean(axis=0)

        return np.stack((fit[0] * y ** 2 + fit[1] * y + fit[2], y)).astype(np.int).T

    def radius_of_curvature(self):
        ym_per_pix = self._image_lane_length / self._meter_per_y_axis
        xm_per_pix = self._land_width / self._meter_per_x_axis

        points = self.generate_points()
        x = points[:, 0]
        y = points[:, 1]
        fit = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        first_derivate = 2 * fit[0] * self._meter_per_y_axis * ym_per_pix + fit[1]
        second_derivate = 2 * fit[0]
        radius = int(((1 + (first_derivate ** 2) ** 1.5) / np.absolute(second_derivate)))
        return radius

    def camera_distance(self):
        xm_per_pix = self._land_width / self._meter_per_x_axis

        points = self.generate_points()
        x = points[np.max(points[:, 1])][0]
        distance = np.absolute((self.width // 2 - x) * xm_per_pix)
        return distance