import numpy as np
import cv2
# image1 = cv2.imread('/home/cougarnet.uh.edu/yzi2/PycharmProjects/cv-homework-2-YuanZi1501040205/2-in000001.jpg')
# image2 = cv2.imread('/home/cougarnet.uh.edu/yzi2/PycharmProjects/cv-homework-2-YuanZi1501040205/2-in001800.jpg')
from scipy.odr import Model, Data, ODR
from scipy.stats import linregress
from sklearn import linear_model
import math
import random
from matplotlib.patches import Ellipse

# total least square linear regression
def orthoregress(x, y):
    """Perform an Orthogonal Distance Regression on the given data,
    using the same interface as the standard scipy.stats.linregress function.
    Arguments:
    x: x data
    y: y data
    Returns:
    [m, c, nan, nan, nan]
    Uses standard ordinary least squares to estimate the starting parameters
    then uses the scipy.odr interface to the ODRPACK Fortran code to do the
    orthogonal distance calculations.
    """
    linreg = linregress(x, y)
    mod = Model(f)
    dat = Data(x, y)
    od = ODR(dat, mod, beta0=linreg[0:2])
    out = od.run()
    return list(out.beta)
def f(p, x):
    """Basic linear regression 'model' for use with ODR"""
    return (p[0] * x) + p[1]

#maximum likelihood linear regression
#   define a function to calculate the log likelihood
def calcLogLikelihood(guess, true, n):
    error = true-guess
    sigma = np.std(error)
    f = ((1.0/(2.0*math.pi*sigma*sigma))**(n/2))* \
        np.exp(-1*((np.dot(error.T,error))/(2*sigma*sigma)))
    return np.log(f)
#   define my function which will return the objective function to be minimized
def myFunction(var):
    #   load my  data
    [x, y] = np.load('myData.npy')
    yGuess = (var[2]*(x**2)) + (var[1]*x) + var[0]
    f = calcLogLikelihood(yGuess, y, float(len(yGuess)))
    return (-1*f)
# %%
class FittingModels:

    def create_dataPoints(self, image1, image2):
        """ Creates a list of data points given two images
        :param image1: first image
        :param image2: second image
        :return: a list of data points
        """
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        points = []
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                points.append([image1[i][j], image2[i][j]])
        return points

    def plot_data(self, data_points):
        """ Plots the data points
        :param data_points:
        :return: an image
        """

        import matplotlib.pyplot as plt
        import numpy as np
        import os

        #convert points' pair to x, y for plotting
        x, y = np.array(data_points).T
        #create figure and read it as image object
        fig = plt.figure()
        plt.plot(x, y, 'bo')
        fig.savefig('plotcache.jpg')
        im = cv2.imread('plotcache.jpg')
        os.remove('plotcache.jpg')
        return im

    def fit_line_ls(self, data_points, threshold):
        """ Fits a line to the given data points using least squares
        :param data_points: a list of data points
        :param threshold: a threshold value (if > threshold, imples outlier)
        :return: a tuple containing the followings:
                    * An image showing the line along with the data points
                    * The thresholded image
                    * A segmented image
        """
        # %%
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        # line_fitting_ls total least square fitting
        #convert points' pair to x, y for fitting
        X, Y = np.array(data_points).T
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)

        num = 0
        den = 0
        for i in range(len(X)):
            num += (X[i] - X_mean) * (Y[i] - Y_mean)
            den += (X[i] - X_mean) ** 2
        m = num / den
        c = Y_mean - m * X_mean

        # Printing coefficients
        print("Coefficients")
        print(m, c)
        # Plotting Values and Regression Line
        max_x = np.max(X)
        min_x = np.min(X)

        # Calculating line values x and y
        x = np.linspace(min_x, max_x, 1000)
        y = c + m * x

        # Ploting Line
        plt.plot(x, y, color='#58b970', label='Slope')
        # Ploting Scatter Points
        plt.scatter(X, Y, c='#ef5423', label='Data')

        plt.show()
        plt.savefig('plotcache.jpg')
        im_line_fitting_ls = cv2.imread('plotcache.jpg')
        os.remove('plotcache.jpg')

        #thresholded_ls
        import math
        #Caculate distance between the point and the line
        #save segmentation pixels based on the threshold
        distance = []
        af_thresholded_data_points = []
        af_thresholded_plot_points = []
        for i in range(n):
            distance.append(abs(m*data_points[i][0]-data_points[i][1]+c)/math.sqrt(m*m+1))
            if distance[i] > threshold:
                af_thresholded_plot_points.append(data_points[i])
                af_thresholded_data_points.append(255)
            else:
                af_thresholded_data_points.append(0)
                pass
        X, Y = np.array(af_thresholded_plot_points).T
        # Ploting Scatter Points
        plt.scatter(X, Y, c='#ef5423', label='Data')
        plt.legend()
        plt.show()
        plt.savefig('plotcache.jpg')
        thresholded_ls = cv2.imread('plotcache.jpg')
        os.remove('plotcache.jpg')

        #segmented_ls
        image1 = cv2.imread('2-in000001.jpg', 0)
        bf_denoise_seg_array = np.reshape(af_thresholded_data_points, (240, 320))
        bf_denoise_seg_array = np.array(bf_denoise_seg_array, dtype=np.uint8)
        # plt.imshow(bf_denoise_seg_array, cmap='gray', vmin=0, vmax=255)
        # plt.show()
        #set morphology kernel = 5*5
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(bf_denoise_seg_array, kernel, iterations=1)
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

        # cv2.imshow('Input', bf_denoise_seg_array)
        # cv2.imshow('Erosion', img_erosion)
        # cv2.imshow('Dilation', img_dilation)
        # cv2.waitKey(0)

        print("LS")
        return (im_line_fitting_ls, thresholded_ls, img_dilation)

    def fit_line_robust(self, data_points, threshold):
        """ Fits a line to the given data points using robust estimators
        :param data_points: a list of data points
        :param threshold: a threshold value (if > threshold, imples outlier)
        :return: a tuple containing the followings:
                    * An image showing the line along with the data points
                    * The thresholded image
                    * A segmented image
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        # # Robust line_fitting_ls: RANSAC
        # # convert points' pair to x, y for fitting
        # X, Y = np.array(data_points).T
        # # ransac fitting
        # # Fit line using robust linear model estimation
        # ransac = linear_model.LinearRegression()
        # ransac.fit(X.reshape(-1,1), Y.reshape(-1, 1))
        # m = ransac.coef_
        # c = ransac.intercept_
        # random.seed(123) # for testing

        # Reshape image to Nx1 array
        X, Y = np.array(data_points).T
        max_iter = 1000
        samples = 2
        m = 0
        c = 0
        inlier_thresh = 2
        best_fit = [0, 0]
        top_inlier_count = 0
        for i in range(max_iter):
            index = random.sample(range(0, X.shape[0]), samples)
            point1 = (int(X[index[0]]), int(Y[index[0]]))
            point2 = (int(X[index[1]]), int(Y[index[1]]))
            # print(point1, point2)
            # find least squared line
            if (point2[0] - point1[0]) == 0:
                continue
            m = (point2[1] - point1[1]) / (point2[0] - point1[0])
            c = point1[1] - m * point1[0]

        # print(m,b)

        # for j in range(x.shape[0]):

        # d = abs(m*x[j] + y[j] + b)/np.sqrt(m**2 + 1**2)

        # print(d.shape)

        # y = mx+b

        # A = m B = 1 C = b

        d = abs(m * X - Y + b) / (np.sqrt(m ** 2 + 1 ** 2))

        # print(d[0:5], "d")

        # print(x[200], y[200])

        # print(d[d>inlier_thresh].size)

        if (d[d < inlier_thresh].size > top_inlier_count):
            best_fit[0] = m
            best_fit[1] = c


        # Printing coefficients
        print("Coefficients")
        print(m, c)
        # Plotting Values and Regression Line
        max_x = np.max(X)
        min_x = np.min(X)

        # Calculating line values x and y
        x = np.linspace(min_x, max_x, 1000)
        y = c + m * x

        # Ploting Line
        plt.plot(x, y, color='#58b970', label='Slope')
        # Ploting Scatter Points
        plt.scatter(X, Y, c='#ef5423', label='Data')

        plt.legend()
        plt.show()
        plt.savefig('plotcache.jpg')
        im_line_fitting_ls = cv2.imread('plotcache.jpg')
        os.remove('plotcache.jpg')

        # thresholded_ls
        import math
        # Caculate distance between the point and the line
        # save segmentation pixels based on the threshold
        distance = []
        af_thresholded_data_points = []
        af_thresholded_plot_points = []
        n = len(X)
        for i in range(n):
            distance.append(abs(m * data_points[i][0] - data_points[i][1] + c) / math.sqrt(m * m + 1))
            if distance[i] > threshold:
                af_thresholded_plot_points.append(data_points[i])
                af_thresholded_data_points.append(255)
            else:
                af_thresholded_data_points.append(0)
                pass
        X, Y = np.array(af_thresholded_plot_points).T
        # Ploting Scatter Points
        plt.scatter(X, Y, c='#ef5423', label='Data')
        plt.legend()
        plt.show()
        plt.savefig('plotcache.jpg')
        thresholded_ls = cv2.imread('plotcache.jpg')
        os.remove('plotcache.jpg')

        #segmented_ls
        image1 = cv2.imread('2-in000001.jpg', 0)
        bf_denoise_seg_array = np.reshape(af_thresholded_data_points, (240, 320))
        bf_denoise_seg_array = np.array(bf_denoise_seg_array, dtype=np.uint8)
        # plt.imshow(bf_denoise_seg_array, cmap='gray', vmin=0, vmax=255)
        # plt.show()
        #set morphology kernel = 5*5
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(bf_denoise_seg_array, kernel, iterations=1)
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

        # cv2.imshow('Input', bf_denoise_seg_array)
        # cv2.imshow('Erosion', img_erosion)
        # cv2.imshow('Dilation', img_dilation)
        # cv2.waitKey(0)
        print("RO")
        return (im_line_fitting_ls, thresholded_ls, img_dilation)

    def fit_gaussian(self, data_points, threshold):
        """ Fits the data points to a gaussian
        :param data_points: a list of data points
        :param threshold: a threshold value (if < threshold, imples outlier)
        :return: a tuple containing the followings:
                    * An image showing the gaussian along with the data points
                    * The thresholded image
                    * A segmented image
        """
        import numpy as np
        import matplotlib.pyplot as plt
        #from scipy.stats import gaussian_kde
        import os

        X, Y = np.array(data_points).T
        # mean_x = np.mean(X)
        # mean_y = np.mean(Y)
        #
        # # Calculate the point density
        # XY = np.vstack([X, Y])
        # Z = gaussian_kde(XY)(XY)
        #
        # # Sort the points by density, so that the densest points are plotted last
        # idx = Z.argsort()
        # X, Y, Z = X[idx], Y[idx], Z[idx]
        #
        # fig, ax = plt.subplots()
        # ax.scatter(X, Y, c=Z, s=50, edgecolor='')
        # plt.show()

        # print(np.mean(x), np.mean(y), "Mean")
        # print(np.std(x), np.std(y), "Stddev")
        cov = np.cov(X, Y, rowvar=0)
        # print(cov)
        eigval, eigvec = np.linalg.eigh(cov)
        # print(eigval, "eigval")
        # print(eigvec, "eigvect")
        order = eigval.argsort()[::-1]
        # print(eigval[order])
        # print(eigvec[:,order])
        eigvec = eigvec[:, order]
        eigval = eigval[order]
        angle = np.degrees(np.arctan2(eigvec[(1, 0)], eigvec[(0, 0)]))
        std = .1

        while True:
            w, h = std * 2 * np.sqrt(eigval)
            # d = (x/(w))**2 + (y/(h))**2
            d = (np.cos(angle * np.pi / 180) * (X - np.mean(X)) + np.sin(angle * np.pi / 180) * (Y - np.mean(Y))) ** 2 / (
                    .5 * w) ** 2
            d += (np.sin(angle * np.pi / 180) * (X - np.mean(X)) - np.cos(angle * np.pi / 180) * (Y - np.mean(Y))) ** 2 / (
                    .5 * h) ** 2
            if (d[d < 1].size / d.size >= threshold):
                break
            std += .1
        # print(std)
        # print(d)
        # print(d[d<1].size, "size")
        thresh_mask = np.zeros((d.shape))
        # print(thresh_mask.size)
        segmentation = np.zeros((d.shape))
        for i in range(d.size):
            if (d[i] >= 1):
                thresh_mask[i] = 1
                segmentation[i] = 255
        # print(thresh_mask)
        outlier_points = np.stack((X * thresh_mask, Y * thresh_mask), 1)
        # print(outlier_points.shape)
        outlier_points[outlier_points == 0] = -1
        w, h = 2 * 1 * np.sqrt(eigval)
        # print(w, "new W")
        # print(h,"new H")
        GA_plot_fig, GA_plot = plt.subplots()
        ellipse683 = Ellipse(xy=(np.mean(X), np.mean(Y)), width=w, height=h, angle=angle,
                             color='blue', linewidth=2)
        ellipse683.set_facecolor('none')
        GA_plot.add_artist(ellipse683)
        w, h = 2 * 2 * np.sqrt(eigval)
        ellipse954 = Ellipse(xy=(np.mean(X), np.mean(Y)), width=w, height=h, angle=angle,
                             color='blue', linewidth=2)
        ellipse954.set_facecolor('none')
        GA_plot.add_artist(ellipse954)
        w, h = 2 * 3 * np.sqrt(eigval)
        ellipse997 = Ellipse(xy=(np.mean(X), np.mean(Y)), width=w, height=h, angle=angle,
                             color='blue', linewidth=2)
        ellipse997.set_facecolor('none')
        GA_plot.add_artist(ellipse997)
        GA_plot.scatter(X, Y, c='r', s=2)
        plt.show()
        plt.savefig('plotcache.jpg')
        GA_plot_img = cv2.imread('plotcache.jpg')
        os.remove('plotcache.jpg')
        GA_thresh_fig, GA_thresh = plt.subplots()

        GA_thresh.scatter(outlier_points[:, 0], outlier_points[:, 1], c='r', s=2)
        plt.show()
        plt.savefig('plotcache.jpg')
        GA_thresh_img = cv2.imread('plotcache.jpg')
        os.remove('plotcache.jpg')
        GA_seg_img = np.reshape(segmentation, (240, 320))

        # Taking a matrix of size 3 as the kernel
        kernel = np.ones((5, 5), np.uint8)
        GA_seg_img = cv2.erode(GA_seg_img, kernel, iterations=1)
        GA_seg_img = cv2.dilate(GA_seg_img, kernel, iterations=1)
        plt.imshow(GA_seg_img)
        plt.show()
        print("GA")
        return (GA_plot_img, GA_thresh_img, GA_seg_img)
