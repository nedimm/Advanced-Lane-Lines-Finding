import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.image_reader import ImageReader
from src.image_plotter import ImagePlotter
from src.threshold_applier import ThresholdApplier

class Camera:
    def __init__(self, image_reader, image_plotter=None, pattern_shape=(9,6)):
        """ Implementation of camera calibration.
        :param ImageReader image_reader: reads images from an regex
        :param ImagePlotter image_plotter: plots images by category as subplots
        :param tuple pattern_shape: calibration pattern shape
        """
        self.image_reader = image_reader
        self.image_plotter = image_plotter
        self.pattern_shape = pattern_shape
        self.objpoints = []  # 3d points in real world
        self.imgpoints = []  # 2d points in image plane
        self.read_mode = image_reader.read_mode
        self.calibrated = False
        self.image_size = 0
        self.camera_matrix = None
        self.distortion_coefficients = None

    def _map_3Dpoints_2Dpoints(self):
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        for name, image in self.image_reader.images():
            if self.image_size == 0:
                self.image_size = image.shape[0:2]
            gray = self.image_reader.get_gray_image(image)
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_shape, None)

            if ret:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                cv2.drawChessboardCorners(image, (self.pattern_shape[0], self.pattern_shape[1]), corners, ret)
                if self.image_plotter is not None:
                    self.image_plotter.add_to_plot(gray, name + " gray", 'gray', True)
                    self.image_plotter.add_to_plot(image, name + " with corners", 'with_corners', False)
            else:
                if self.image_plotter is not None:
                    self.image_plotter.add_to_plot(image, name + " failed", 'failed', True)

    def calibrate(self):
        self._map_3Dpoints_2Dpoints()
        if len(self.objpoints) > 0:
            _, self.camera_matrix, self.distortion_coefficients, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.image_size, None, None)
            self.calibrated = True

    def undistort(self, image):
        if self.calibrated:
            return cv2.undistort(image, self.camera_matrix, self.distortion_coefficients, None, self.camera_matrix)
        else:
            return image

    def perspective_transform(self, image, source_corners, destination_corners):
        height = image.shape[0]
        width = image.shape[1]
        M = cv2.getPerspectiveTransform(source_corners, destination_corners)
        warped = cv2.warpPerspective(image, M, (width, height))
        unwrap_m = cv2.getPerspectiveTransform(destination_corners, source_corners)
        return (warped, unwrap_m)

    def get_region_of_interest(self, image):
        height = image.shape[0]
        width = image.shape[1]
        # region of interest points in the source image
        s1 = [100, height]
        s2 = [width // 2 - 50, height * 0.625]
        s3 = [width // 2 + 90, height * 0.625]
        s4 = [width - 50, height]
        src = np.float32([s1, s2, s3, s4])

        # region of interest points in the destination image
        d1 = [100, height]
        d2 = [100, 0]
        d3 = [width - 50, 0]
        d4 = [width - 50, height]
        dst = np.float32([d1, d2, d3, d4])
        return src, dst


def test_calibration():
    # Please uncomment lines of interest to see the images
    ir = ImageReader(read_mode='RGB')
    ip = ImagePlotter()
    cc = Camera(ir, ip)
    cc.calibrate()

    # show distorted images
    #ip.plot('gray', 3)

    # show images with recognized corners
    #ip.plot('with_corners', 3)

    # show images where mapping points on the 3d image to 2D image did not work
    #ip.plot('failed', 3)


    # Visualize undistortion
    test_image = cv2.imread('../camera_cal/calibration1.jpg')
    undistorted_image = cc.undistort(test_image)
    ip.add_to_plot(test_image, 'image to undistort', 'undistort', False)
    ip.add_to_plot(undistorted_image, 'undistorted image', 'undistort', False)
    ip.plot('undistort', 2)

    plt.show()

def test_warp():
    ip = ImagePlotter(images_to_show_count=15)
    ir = ImageReader(read_mode='RGB', regex='../test_images/test*.jpg')
    ta = ThresholdApplier()
    for image_file in ir.images():
        ip.add_to_plot(image_file.image, image_file.name, 'input', False)

        #height = image_file.image.shape[0]
        #width = image_file.image.shape[1]
        ## region of interest points in the source image
        #s1 = [100, height]
        #s2 = [width // 2 - 50, height * 0.625]
        #s3 = [width // 2 + 90, height * 0.625]
        #s4 = [width-50, height]
        #src = np.float32([s1, s2, s3, s4])

        # region of interest points in the destination image
        #d1 = [100, height]
        #d2 = [100, 0]
        #d3 = [width - 50, 0]
        #d4 = [width - 50, height]
        #dst = np.float32([d1, d2, d3, d4])

        cc = Camera(ir, ip)
        src, dst = cc.get_region_of_interest(image_file.image)

        warped, unwarp_m = cc.perspective_transform(image_file.image, src, dst)
        ip.add_to_plot(warped, image_file.name + ' warped', 'warped', False)

        ip.add_to_plot(image_file.image, image_file.name, 'side', False)
        ip.add_to_plot(warped, image_file.name + ' warped', 'side', False)

        #pts = np.int32([s1, s2, s3, s4])
        pts = np.int32(src)
        cv2.polylines(image_file.image, [pts], 0, color=[255, 0, 0], thickness=10)
        #dst = np.int32([d1, d2, d3, d4])
        #cv2.polylines(image_file.image, [dst], 0, color=[0, 255, 0], thickness=10)

        threshold_applied = ta.apply_combined_thresholds(warped)
        ip.add_to_plot(threshold_applied, image_file.name + " threshold applied", 'side', True)

    ip.plot('side', 3)






if __name__ == "__main__":
    #test_calibration()
    test_warp()