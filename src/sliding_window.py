import numpy as np

class SlidingWindow(object):
    """Represents a window where we want to find 'hot' Pixels associated with Lane Lines"""

    def __init__(self, y_low, y_high, x_center, margin=100, minpix=50):
        self.x = x_center
        self.x_mean = x_center
        self.y_high = y_high
        self.y_low = y_low
        self.margin = margin
        self.minpix = minpix

    def nonzero_pixels_indices(self, nonzero, win_x=None):
        if win_x is not None:
            self.x = win_x

        win_inds = ((nonzero[0] >= self.y_high) &
                    (nonzero[0] < self.y_low) &
                    (nonzero[1] >= self.x - self.margin) &
                    (nonzero[1] < self.x + self.margin)).nonzero()[0]

        # Find the x mean value
        if len(win_inds) > self.minpix:
            self.x_mean = np.int(np.mean(nonzero[1][win_inds]))
        else:
            self.x_mean = self.x

        return win_inds

    def vertices(self):
        """Tuple of boundary rectangle vertices"""
        x1 = self.x - self.margin
        y1 = self.y_high
        x2 = self.x + self.margin
        y2 = self.y_low
        return ((x1, y1), (x2, y2))