#**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./undistort_image1.jpg "Undistorted"
[image2]: ./straight_lines1.jpg "Road Transformed"
[image3]: ./binary_image1.jpg "Binary Example"
[image4]: ./warp_image1.jpg "Warp Example"
[image5]: ./lane_fit.jpg "Fit Visual"
[image6]: ./result_image2.jpg "Output"
[video1]: ./test_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

[Here](https://github.com/smashkoala/CarND-Advanced-Lane-Lines/blob/master/writeup.md) in GitHub, you can find my writeup.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `prepare_calibration()` and `undistort_images()` in `lane_detect.py`.  

1) prepare_calibration()  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

2) undistort_images()  
I then use the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I apply this distortion correction to the test image using the `cv2.undistort()` function and obtain this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The following is a distortion-corrected image, which is used in the explanation of the pipeline.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 78 through 103 in `pipeline()` of `lane_detect.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included in a function called `warp_image()`, which appears in lines 105 through 119 in the file `lane_detect.py`.  The `warp_image()` function takes an image (`img`) as an input. Source (`src`) and destination (`dst`) points are given through a global variable `corners`.  I hardcoded the source and destination points. These points are manually identified by visually checking the coordinates of the transforming area in a drawing tool.

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 200, 720      | 200,  720     |
| 580, 460      | 200,  0       |
| 698, 460      | 1060, 0       |
| 1060,720      | 1060, 720     |

I verified that my perspective transform was working as expected by visually checking warped images which has straight lane lines. The following is an example image of the warped images.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane detection process takes place in `find_lanes()` function in `lane_detect.py` (code line from 135 to 224). It takes a histogram of the bottom half of the image to find the peaks of the left and right lines. These become the starting points for the left and right lines.
With these starting points, it creates windows on both left and right lines. The width of the windows is 200 pix, and the height of the windows is 80 pix. Within these windows, it collects all positions which are none-zero pixels. It checks if 50 pix or more of none-zero pixels are included in one window. If the window contains 50 pix or more, the center of the window is changed to the mean of none-zero pixel positions when the window slides upward.
With these pixel positions, it fits a second order polynomial to find the lane lines. The following image show how this process works.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The second order polynomial of real world space is calculated line 221-222 of `lane_detect.py`. Based on this polynomial, the curvature is calculated at line 89 of `line.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step from lines 253 through 287 in `lane_detect.py` (in the function `draw_image()`).  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For the pipeine processing, in order to see if detected lines are parallel , I checked the distance between left and right lines.(Line 294-296 of `lane_detect.py`). When the distance is not within the range (pixel length < 800 or > 950), both left and right line data are discarded without checking them individually, and using the mean of previous n data instead. This might be improved in order to keep either of valid line data because most of the time data is discarded when only one of the lines is not well detected.
