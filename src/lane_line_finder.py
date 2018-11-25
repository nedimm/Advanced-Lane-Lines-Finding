import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from src.sliding_window import SlidingWindow
from src.lane_line import LaneLine
from src.image_reader import ImageReader
from src.image_plotter import ImagePlotter
from src.camera import Camera
from src.threshold_applier import ThresholdApplier


class LaneLineFinder(object):
    """Detects the lane lines in images or a video stream"""

    def __init__(self, threshold_applier, calibrated_camera, region_of_interest, first_frame, windows_cnt=9):
        self.threshold_applier = threshold_applier
        self.camera = calibrated_camera
        self.region_of_interest = region_of_interest
        self.height = first_frame.shape[0]
        self.width = first_frame.shape[1]
        self.windows_cnt = windows_cnt
        self.left_lane = None
        self.right_lane = None
        self.left_wins = []
        self.right_wins = []
        self._init_lanes(first_frame)
        self._sliding_window_frame = None
        self._overlay = None

    def _init_lanes(self, frame):
        thresholds_applied = self.threshold_applier.apply_combined_thresholds(frame)
        warped, unwrap_m = self.camera.perspective_transform(thresholds_applied, self.region_of_interest[0], self.region_of_interest[1])

        frame_histogram = np.sum(warped[int(self.height / 2):, :], axis=0)
        nonzero = warped.nonzero()

        left_lane_indexes = np.empty([0], dtype=np.int)
        right_lane_indexes = np.empty([0], dtype=np.int)
        window_height = int(self.height / self.windows_cnt)

        for i in range(self.windows_cnt):
            if len(self.left_wins) > 0:
                l_x_center = self.left_wins[-1].x
                r_x_center = self.right_wins[-1].x
            else:
                l_x_center = np.argmax(frame_histogram[:self.width // 2])
                r_x_center = np.argmax(frame_histogram[self.width // 2:]) + self.width // 2

            left_win = SlidingWindow(y_low=self.height - i * window_height,
                                     y_high=self.height - (i + 1) * window_height,
                                     x_center=l_x_center)

            right_win = SlidingWindow(y_low=self.height - i * window_height,
                                      y_high=self.height - (i + 1) * window_height,
                                      x_center=r_x_center)

            left_lane_indexes = np.append(left_lane_indexes, left_win.nonzero_pixels_indices(nonzero), axis=0)
            right_lane_indexes = np.append(right_lane_indexes, right_win.nonzero_pixels_indices(nonzero), axis=0)
            self.left_wins.append(left_win)
            self.right_wins.append(right_win)

        self.left_lane = LaneLine(nonzero[1][left_lane_indexes], nonzero[0][left_lane_indexes],
                                  self.height, self.width)
        self.right_lane = LaneLine(nonzero[1][right_lane_indexes], nonzero[0][right_lane_indexes],
                                   self.height, self.width)

    def _find_lane_pixels_for_each_window(self, frame, windows):
        indices = np.empty([0], dtype=np.int)
        nonzero = frame.nonzero()
        win_x = None

        for win in windows:
            indices = np.append(indices, win.nonzero_pixels_indices(nonzero, win_x), axis=0)
            win_x = win.x_mean

        return (nonzero[1][indices], nonzero[0][indices])

    def _lane_overlay(self, image, unwrap_m=None):
        overlay = np.zeros_like(image).astype(np.uint8)
        points = np.vstack((self.left_lane.generate_points(),
                            np.flipud(self.right_lane.generate_points())))
        cv2.fillPoly(overlay, [points], (0, 255, 0))

        if unwrap_m is not None:
            overlay = cv2.warpPerspective(overlay, unwrap_m, (image.shape[1], image.shape[0]))

        alpha = 0.6
        return cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)

    def _info_overlay(self, frame):
        if len(frame.shape) == 2:
            image = np.dstack((frame, frame, frame))
        else:
            image = frame

        for win in self.left_wins:
            vertices = win.vertices()
            cv2.rectangle(image, vertices[0], vertices[1], (1.0, 1.0, 0), 2)

        for win in self.right_wins:
            vertices = win.vertices()
            cv2.rectangle(image, vertices[0], vertices[1], (1.0, 1.0, 0), 2)

        cv2.polylines(image, [self.left_lane.generate_points()], False, (1.0, 1.0, 0), 2)
        cv2.polylines(image, [self.right_lane.generate_points()], False, (1.0, 1.0, 0), 2)

        return image * 255

    def _radius_of_curvature(self):
        radius = int(np.average([self.left_lane.radius_of_curvature(),
                                 self.right_lane.radius_of_curvature()]))
        return radius

    def _draw_text(self, frame, text, x, y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    def run(self, frame):
        thresholds_applied = self.threshold_applier.apply_combined_thresholds(frame)
        warped, unwrap_m = self.camera.perspective_transform(thresholds_applied, self.region_of_interest[0], self.region_of_interest[1])

        (left_x, left_y) = self._find_lane_pixels_for_each_window(warped, self.left_wins)
        self.left_lane.fit_points(left_x, left_y)
        (right_x, right_y) = self._find_lane_pixels_for_each_window(warped, self.right_wins)
        self.right_lane.fit_points(right_x, right_y)

        thresholds_applied = self.threshold_applier.apply_stacked_thresholds(frame)
        warped, unwrap_m = self.camera.perspective_transform(thresholds_applied, self.region_of_interest[0], self.region_of_interest[1])
        info_overlay = self._info_overlay(warped)

        warped, unwrap_m = self.camera.perspective_transform(frame, self.region_of_interest[0], self.region_of_interest[1])
        top_overlay = self._lane_overlay(warped)
        self._sliding_window_frame = info_overlay
        info_overlay = cv2.resize(info_overlay, (0, 0), fx=0.3, fy=0.3)
        top_overlay = cv2.resize(top_overlay, (0, 0), fx=0.3, fy=0.3)
        self._overlay = top_overlay
        frame[:250, :, :] = frame[:250, :, :] * 0.4
        (height, width, _) = info_overlay.shape
        frame[25:25 + height, 25:25 + width, :] = info_overlay
        frame[25:25 + height, 25 + 25 + width:25 + 25 + width + width, :] = top_overlay
        text_x = 25 + 25 + width + width + 25

        lane_width = self.left_lane.camera_distance() + self.right_lane.camera_distance()
        off_center = self.right_lane.camera_distance() - lane_width / 2
        self._draw_text(frame, 'Radius of curvature:  {} m'.format(self._radius_of_curvature()), text_x, 80)
        self._draw_text(frame, 'Distance of center:    {:.1f} m'.format(off_center), text_x, 140)

        return self._lane_overlay(frame, unwrap_m)

def create_video():
    ir = ImageReader(read_mode='RGB')
    cc = Camera(ir, None)
    cc.calibrate()
    ta = ThresholdApplier()

    output_video_name = '../output_videos/project_video_result.mp4'
    input_video = VideoFileClip("../project_video.mp4")

    image = input_video.get_frame(0)
    undistorted = cc.undistort(image)
    llf = LaneLineFinder(ta, cc, (cc.get_region_of_interest(image)), undistorted)
    output_video = input_video.fl_image(llf.run)
    output_video.write_videofile(output_video_name, audio=False)

def test_with_images():
    ir = ImageReader(read_mode='RGB')
    ip = ImagePlotter(images_to_show_count=10)
    cc = Camera(ir, ip)
    cc.calibrate()
    ir.regex = '../test_images/test*.jpg'
    ir.set_image_names()
    ta = ThresholdApplier()

    for image_file in ir.images():
        image = image_file.image
        ip.add_to_plot(image, image_file.name, 'out', False)
        undistorted = cc.undistort(image)
        llf = LaneLineFinder(ta, cc, (cc.get_region_of_interest(image)), image)
        result = llf.run(undistorted)
        # ip.add_to_plot(llf._sliding_window_frame, image_file.name + ' sliding windows', 'out', False)
        # ip.add_to_plot(result, image_file.name + 'result', 'out', False)
        ip.add_to_plot(llf._overlay, image_file.name + ' overlay', 'out', False)
    ip.plot('out', 2)

if __name__ == "__main__":
    #test_with_images()
    create_video()
