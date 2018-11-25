## Advanced Lane Finding
[![Advanced Lane Finding on the Road](https://i.imgur.com/6gJ57WS.png)](https://www.youtube.com/watch?v=1BTBnllGh1o "Advanced Lane Finding")
[More results on Youtube](https://www.youtube.com/watch?v=F7gluNuSx50&list=PL06vO3TcKwfYCAyu5FBqnhxylDzH1chP2)


In this Advanced Lane Finding project, we will apply computer vision techniques to detected road lanes in a video from a front-facing camera on a car. The video itself was supplied by Udacity. The project is done in Python and OpenCV.


The Goals
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform of a region of interest on an image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing the pipeline on single frames. The output from each stage of the pipeline are saved in the folder called `output_images`. The pipline is tested on the video called `project_video.mp4`.  

Camera Calibration
---
Real cameras use curved lenses to form an image, and light rays often bend a little too much or too little at the edges of these lenses. This creates an effect that distorts the edges of images, so that lines or objects appear more or less curved than they actually are. This is called radial distortion, and it’s the most common type of distortion.

Another type of distortion, is tangential distortion. This occurs when a camera’s lens is not aligned perfectly parallel to the imaging plane, where the camera film or sensor is. This makes an image look tilted so that some objects appear farther away or closer than they actually are.

Also, the distortion changes the size and shapes of objects in an image. In order to calibrate the camera we take some pictures of known shapes and detect and correct any distortion errors. In this project we'll use pictures of a chessboard. A cheesboard is great for calibration because its regular high contrast pattern makes it easy to detect automatically, and we know what an undistorted flat chessboard looks like. So, if we use our camera to take multiple pictures of a cheesboard against a flat surface, then we'll be able to detect any distortion by looking at the difference between the apparent size and the shape of the squares in these images, and the size and shape that they actually are. Then we'll use that information to calibrate our camera, create a transform that maps these distorted points to undistorted points and finaly undisrtort any images. 
Let's see how to do this using Python and OpenCV.

The distorted images look like this:
![distorted images](https://i.imgur.com/RoaHc3Q.png "output_images/figure1-distorted_images.png")

We will first use the OpenCV functions `findChessboardCorners()` and `drawChessboardCorners()` to automatically find and draw corners in an image of a chessboard pattern. The implementation of the function that maps the points from 3d image to the 2d image plane is show below.
```python
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
```
The found chessboard corners drawn on the distorted images look like this:
![distorted images_with_corners](https://i.imgur.com/K1xke82.png "output_images/figure2-distorted_images_with_corners.png")

After mapping the points from 3D image to the 2D image, the calibration of the camera is just calling the OpenCV `calibrateCamera()` function:
```python
    def calibrate(self):
        self._map_3Dpoints_2Dpoints()
        if len(self.objpoints) > 0:
            _, self.camera_matrix, self.distortion_coefficients, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.image_size, None, None)
            self.calibrated = True
```

Distortion Correction
---

Now that the camera is calibrated we can undistort any image by calling the `undistort()` function:
```python
    def undistort(self, image):
        if self.calibrated:
            return cv2.undistort(image, self.camera_matrix, self.distortion_coefficients, None, self.camera_matrix)
        else:
            return image
```
Here is a an example of distorted and undistorted image using the process described above:
![distorted / undistorted image](https://i.imgur.com/82XuMV2.png "output_images/figure4-distorted_undistored_images.png")

Perspective Transform
---
In an image, perspective is the phenomenon where an object appears smaller the farther awas it is from a viewpoint like a camera, and parallel lines appear to converge to a point. In an image of the road, the lane looks smaller and smaller the farther awas it gets from the camera, and the background scenery also appears smaller than the trees closer to the camera in the foreground. Mathematically, we can characterize perspective by saying that, in real world coordinates x,y and z the greater the magnitude of an objects z coordinate, or distance from the camera, the smaller it will appear in a 2D image. A perspective transform uses this information to transform an image. It essentially transforms the apparent z coordinate of the object points, which in turn changes that object's 2D image representation. A perspective transform warps the image and effectively drags points towards or pusches them away from the camera to change the apparent perspective. For example, to change this into a bird's eye view scene, we can apply a perspective transform that zooms in on the farther away objects. This is really useful, because finding the curvature of a lane is easier to perform on a bird's eye view of an image. 
Our next step then is to isolate a region of interest and convert that part of the image into a birds-eye-view by calling the OpenCV functions as shown in the code below. 
```python
    def perspective_transform(self, image, source_corners, destination_corners):
        height = image.shape[0]
        width = image.shape[1]
        M = cv2.getPerspectiveTransform(source_corners, destination_corners)
        warped = cv2.warpPerspective(image, M, (width, height))
        unwrap_m = cv2.getPerspectiveTransform(destination_corners, source_corners)
        return (warped, unwrap_m)
```
The `source_corners` form the region of interest shown in red on the image below. 
![warped images](https://i.imgur.com/SGwFfWx.png "output_images/Figure_5_warped_images.png")
The warped image can now be used to detect lane lines.

Creating a thresholded binary image
---
In the first introduction project "Finding Lane Lines on the Road", we used Canny edge detection to find pixels that were likely to be part of a line in an image. Canny is great at finding all possible lines in an image, but for lane detection, this gave us a lot of edges on scenery, and cars, and other objects that we ended up discarding.
Realistically, with lane finidng we know ahead of time that the lines we are looking for are close to vertical. In this project we can use gradients in a smarter way to detect steep edges that are more likely to be lanes in the first place. With Canny, we were actually taking a derivative with respect to X and Y in the process of finding edges. In this project we will also start with applying Sobel operator to an image as a way of taking the derivative of the image in the X or Y direction. We then calculate absolute value of X and Y derivatives and convert the absolute value image to 8 bit. We then return pixels within a treshold:

```python
    def _get_gradient_absolute_value_mask(self, image, dx, dy, threshold, sobel_ksize=3):
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=sobel_ksize))
        return self._get_mask(sobel, threshold)

    def _get_mask(self, image, threshold):
        scaled_image = np.uint8(255 * image / np.max(image))
        mask = np.zeros_like(scaled_image)
        mask[(scaled_image >= threshold[0]) & (scaled_image <= threshold[1])] = 1
        return mask
```
We then apply a threshold to the overall magnitude of the gradient, in both x and y. 
```python
    def _get_gradient_magnitude_mask(self, image, threshold, sobel_ksize=3):
        x, y = self._get_gradients(image, sobel_ksize)
        magnitude = np.sqrt(x ** 2 + y ** 2)
        return self._get_mask(magnitude, threshold)

    def _get_gradients(self, image, sobel_ksize):
        x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        return x, y
```
In the case of lane lines, we're interested only in edges of a particular orientation. So now we can experiment with the direction, or orientation, of the gradient. The direction of the gradient is simply the inverse tangent (arctangent) of the Y gradient divided by the X gradient:
```python
def _get_gradient_direction_mask(self, image, threshold, sobel_ksize=3):
    x, y = self._get_gradients(image, sobel_ksize)
    direction_mask = np.arctan2(np.absolute(y), np.absolute(x))
    return self._get_mask(direction_mask, threshold)
```
 To isolate lane-line pixels we can experiment by using various aspects of gradient measurements (x, y, magnitude, direction) and also color selection.
 ```python
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
 ```
 Here is an example of stacking and combining color and gradient thresholds:
 ![thresholds applied](https://i.imgur.com/1n2iAfp.png "output_images/Figure_6_thresholds.png")
 
 Detect Lane Pixels and Find the Lane Boundary
 ---
 After applying calibration, thresholding, and a perspective transform to a road image, we have a binary image where the lane lines stand out. However, we still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

Plotting a histogram of where the binary activations occur across the image is one potential solution for this. 
```python
frame_histogram = np.sum(warped[int(self.height / 2):, :], axis=0)
```
With this histogram we are adding up the pixel values along each column in the image. In our thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. We can use that as a starting point for where to search for the lines. From that point, we can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.
Now that we have found all our pixels belonging to each line through the sliding window method, we fit a polynomial to the line.
```python
def fit_points(self, x, y):
    points = len(y) > 0 and (np.max(y) - np.min(y)) > self.heigth * 0.625
    no_coef = len(self.coef) == 0

    if points or no_coef:
        self.coef.append(np.polyfit(y, x, 2))
```

![sliding windows](https://i.imgur.com/FzkNWC5.png "output_images/figure_8_sliding_windows.png")

Determine the Curvature of the Lane 
---
The radius of curvature at any point `x` of the function `x=f(y)` is given as follows:

![radius](https://i.imgur.com/5CRVRmw.png)
In the case of the second order polynomial above, the first and second derivatives are:
![derivatives](https://i.imgur.com/UWtS5WA.png)
So, the equation for radius of curvature becomes:

![radius](https://i.imgur.com/Cl62Xq3.png)

In Python it was implemented as follows:

```python
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
```

Warp the Detected Lane Boundaries Back onto the Original Image
---
The warped image with an overlay is shown in the next figure:
![overlay](https://i.imgur.com/R6CAO8P.png "output_images/figure_9_overlay.png")
Next, we warp back the part of the image onto the original image.

Visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
---
The final result with the and overlay on the top is presented on the figure below:

![final result](https://i.imgur.com/9m7c5hd.png "output_images/Figure_10_final.png")

Discussion
---
As it can be seen from the video shown in the Jupyter notebook, with this approach we can successfully detect the lane lines on a flat road without elevation and properly marked.
In more challenging situations with an elevation in any direction or a road not being properly marked we could expect problems. As in the previous project, we have here also thresholds that need to be estimated manually. One possible improvement would be if we could use machine learning algorithms to automatically determine the thresholds.