# Define a class to receive the characteristics of each line detection
import numpy as np

class Line():

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False #OK
        # x values of the last n fits of the line
        self.recent_xfitted = np.empty((0, 720), float) #OK
        #polynomials coeffcient of the last n fits of the line
        self.recent_xfit = np.empty((0, 3), float)

        self.recent_xfit_cr = np.empty((0,3), float)

        #average x values of the fitted line over the last n iterations
        self.bestx = None #Cannot see how to average x values
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        self.best_fit_cr = None

        #polynomial coefficients for the most recent fit
        self.current_xfitted = None #OK
        #radius of curvature of the line in some units
        self.current_fit = None

        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') #OK
        # #x values for detected line pixels
        #self.allx = None #Do I need to use this?
        # #y values for detected line pixels
        # self.ally = None #Do I need to use this?

        self.count = 0

    def reset(self):
        self.count = 0
        self.detected = False
        self.recent_xfitted = np.empty((0, 720), float) #OK
        self.recent_xfit = np.empty((0, 3), float)
        self.recent_xfit_cr = np.empty((0, 3), float)
        self.current_xfitted = None
        self.current_fit = None
        self.diffs = np.array([0,0,0], dtype='float') #OK

    def update_lane_data(self, line_fit, line_fitx, line_fit_cr, ploty):
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        max_count = 10

        y_eval = np.max(ploty)
        #curverad = ((1 + (2*line_fit_cr[0]*y_eval*ym_per_pix + line_fit_cr[1])**2)**1.5) / np.absolute(2*line_fit_cr[0])
        #if diff of coefficient is too large, ignore the data
        if self.count > 0:
            self.diffs = line_fit - self.recent_xfit[-1]
            print("self.diffs")
            print(self.diffs)
            print("-------------------")

        #Latest data is NOT OK.
        if (self.diffs > 50).any():
            print(" > 50")
            self.detected = False
            return self.best_fit

        else: #Latest data is OK.
            print("Data is OK")
            self.detected = True
            if self.count > max_count:
                np.delete(self.recent_xfit, 0, 0)
                np.delete(self.recent_xfitted, 0, 0)
            else:
                self.count += 1

            self.recent_xfit = np.append(self.recent_xfit, line_fit.reshape(1,3), axis=0)
            self.recent_xfit_cr = np.append(self.recent_xfit_cr, line_fit_cr.reshape(1,3), axis=0)
            self.recent_xfitted = np.append(self.recent_xfitted, line_fitx.reshape(1, 720), axis=0)
            self.current_fit = line_fit
            self.current_xfitted = line_fitx
            self.bestx = np.mean(self.recent_xfitted, axis = 0)
            self.best_fit = np.mean(self.recent_xfit, axis = 0)
            self.best_fit_cr = np.mean(self.recent_xfit_cr, axis = 0)
            self.radius_of_curvature = ((1 + (2*self.best_fit_cr[0]*y_eval*ym_per_pix + self.best_fit_cr[1])**2)**1.5) / np.absolute(2*self.best_fit_cr[0])
            self.line_base_pos = abs(640.0-self.bestx[-1])*xm_per_pix
            print("Base position = %3f" % self.line_base_pos)

        return self.best_fit
