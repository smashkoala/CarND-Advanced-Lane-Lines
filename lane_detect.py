import pickle
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from line import Line

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

DEBUG = False

def prepare_calibration():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            #Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

        cv2.destroyAllWindows()

    calib_pickle = {}
    calib_pickle["objpoints"] = objpoints
    calib_pickle["imgpoints"] = imgpoints
    pickle.dump( calib_pickle, open( "wide_dist_pickle.p", "wb" ))
    return objpoints, imgpoints

def undistort_images(objpoints, imgpoints, img):
    # Test undistortion on an image
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
#    cv2.imwrite('calibration_wide/test_undist.jpg',dst)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "camera_calib_pickle.p", "wb" ) )
    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # Visualize undistortion
    # cv2.imshow('img', img)
    # cv2.waitKey(1000)
    # cv2.imshow('dst', dst)
    # cv2.waitKey(1000)
    return dst

# Edit this function to create your own pipeline.
# AP: Update this pipeline to provide much better detection
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    combined = np.zeros_like(scaled_sobel)
    combined[((sxbinary == 1) | (s_binary == 1))] = 1
    return combined

def warp_image(img, corners):
     # Grab the image shape
    img_size = (img.shape[1], img.shape[0])

    # For source points I'm grabbing the outer four detected corners
    src = np.float32([corners[0], corners[1], corners[-1], corners[-2]])
#    print(src)
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = np.float32([[corners[0][0], img_size[1]], [corners[0][0], 0],
                                 [corners[2][0], 0],
                                 [corners[2][0], img_size[1]]])
#    print(dst)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    # Return the resulting image and matrix
    return warped

def unwarp_image(img, corners):
    img_size = (img.shape[1], img.shape[0])

    # For source points I'm grabbing the outer four detected corners
    dst = np.float32([corners[0], corners[1], corners[-1], corners[-2]])
#    print(src)
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    src = np.float32([[corners[0][0], img_size[1]], [corners[0][0], 0],
                                 [corners[2][0], 0],
                                 [corners[2][0], img_size[1]]])
#    print(dst)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    unwarped = cv2.warpPerspective(img, M, img_size)

    return unwarped

def find_lanes(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # if DEBUG:
    #     fig = plt.imshow(out_img)
    #     plt.plot(left_fitx, ploty, color='yellow')
    #     plt.plot(right_fitx, ploty, color='yellow')
    #     plt.xlim(0, 1280)
    #     plt.ylim(720, 0)
    #     plt.show()

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    return left_fit, right_fit, left_fit_cr, right_fit_cr

def find_lane_continuous(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    return left_fit, right_fit, left_fit_cr, right_fit_cr

def measure_curvature(left_fit, right_fit, left_fit_cr, right_fit_cr):
    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    # quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
    # # For each y position generate random x position within +/-50 pix
    # # of the line base position in each case (x=200 for left, and x=900 for right)
    # leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
    #                               for y in ploty])
    # rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51)
    #                                 for y in ploty])
    #
    # leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    # rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


    # Fit a second order polynomial to pixel positions in each fake lane line
    # left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Plot up the fake data
    #mark_size = 3
    # plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
    # plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
    if DEBUG:
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(left_fitx, ploty, color='green', linewidth=3)
        plt.plot(right_fitx, ploty, color='green', linewidth=3)
        plt.gca().invert_yaxis() # to visualize as we do the images
        plt.show()

    y_eval = np.max(ploty)
    # left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    # right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    # print(left_curverad, right_curverad)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

def draw_image(original_img, binary_warped, left_fit, right_fit):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = unwarp_image(color_warp, corners)

    # Combine the result with the original image
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    curvature = (left_lane.radius_of_curvature + right_lane.radius_of_curvature) / 2
    text1 = 'Lane curvature = %4.2f m' % curvature

    car_position = (left_lane.line_base_pos - right_lane.line_base_pos / 2)
    if car_position < 0:
        text2 = 'Vehicle is %1.2f m right of the center' % abs(car_position)
    else:
        text2 = 'Vehicle is %1.2f m left of the center' % abs(car_position)
    font = cv2.FONT_HERSHEY_COMPLEX
    font_size = 1.0
    color=(0,0,0)
    result = cv2.putText(result, text1, (10, 50), font, font_size, color, 4)
    result = cv2.putText(result, text2, (10, 100), font, font_size, color, 4)

    if DEBUG:
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()

    return result

def sanity_check(left_fit, right_fit, left_fit_cr, right_fit_cr, ploty):
    global notrack_cnt

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    diff =  right_fitx - left_fitx
    #print(diff)
    if (diff < 800).any() or (diff > 950).any():
        print("Parallel NOK: notrackcnt = %d" % notrack_cnt)
        left_lane.detected = False
        right_lane.detected = False
        notrack_cnt += 1
    else:
        print("Parallel OK")
        left_lane.update_lane_data(left_fit, left_fitx, left_fit_cr, ploty)
        right_lane.update_lane_data(right_fit, right_fitx, right_fit_cr, ploty)
        if left_lane.detected is False or right_lane.detected is False:
            print("update_lane_data NOK: notrack_cnt = %d", notrack_cnt)
            notrack_cnt += 1
        else:
            notrack_cnt = 0

    if notrack_cnt == 10:
        left_lane.reset()
        right_lane.reset()
        notrack_cnt = 0
    return left_lane.bestx, right_lane.bestx

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip


def pipeline_lanes(image):
    result = undistort_images(objpoints, imgpoints, image)
    result2 = pipeline(result)
    result3 = warp_image(result2, corners)
    if left_lane.detected and right_lane.detected:
        print("Continuous")
        left, right, left_cr, right_cr = find_lane_continuous(result3, left_lane.current_fit, right_lane.current_fit)
    else:
        print("All scanning")
        left, right, left_cr, right_cr = find_lanes(result3)
    ploty = np.linspace(0, result3.shape[0]-1, result3.shape[0] )
    sanity_check(left, right, left_cr, right_cr, ploty)
#    measure_curvature(left_lane.best_fit, right_lane.best_fit, left_cr, right_cr)
    data = draw_image(result, result3, left_lane.best_fit, right_lane.best_fit)
    return data

def process_image(image):
    result = pipeline_lanes(image)
    return result

def debug():
    images = glob.glob('test_images/straight_line*.jpg')
    images += glob.glob('test_images/test*.jpg')
    for image_name in images:
        img = cv2.imread(image_name)
        result = undistort_images(objpoints, imgpoints, img)
        result2 = pipeline(result)
        result3 = warp_image(result2, corners)
        if left_lane.detected and right_lane.detected:
            print("Continuous")
            left, right, left_cr, right_cr = find_lane_continuous(result3, left_lane.current_fit, right_lane.current_fit)
        else:
            print("All scanning")
            left, right, left_cr, right_cr = find_lanes(result3)

        ploty = np.linspace(0, result3.shape[0]-1, result3.shape[0] )
        sanity_check(left, right, left_cr, right_cr, ploty)
    #    measure_curvature(left_lane.best_fit, right_lane.best_fit, left_cr, right_cr)
        print("Left lane best fit")
        print(left_lane.best_fit)
        print("Right lane best fit")
        print(right_lane.best_fit)
        data = draw_image(result, result3, left_lane.best_fit, right_lane.best_fit)

#objpoints, imgpoints = prepare_calibration()
#test_images(objpoints, imgpoints)

dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]
corners = [[200, 720], [580, 460], [1060, 720], [698, 460]]
left_lane = Line()
right_lane = Line()
notrack_cnt = 0

output = 'test_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
clip = clip1.fl_image(process_image)
clip.write_videofile(output, audio=False)

#debug()

#Plot the result
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
# f.tight_layout()
#
# ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# ax1.set_title('Original Image', fontsize=20)
#
# ax2.imshow(result, cmap='gray')
# ax2.set_title('Pipeline Result', fontsize=20)
#
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#
# ax3.imshow(result2, cmap='gray')
# ax3.set_title('Warped image', fontsize=20)
#
# plt.show()
