import numpy as np
import cv2
# image1 = cv2.imread('/home/cougarnet.uh.edu/yzi2/PycharmProjects/cv-homework-2-YuanZi1501040205/2-in000001.jpg')
# image2 = cv2.imread('/home/cougarnet.uh.edu/yzi2/PycharmProjects/cv-homework-2-YuanZi1501040205/2-in001800.jpg')

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
        import io
        from PIL import Image
        import matplotlib.pyplot as plt
        #convert points' pair to x, y for plotting
        x, y = np.array(data_points).T
        #create figure and read it as image object
        plt.figure()
        plt.plot(x, y, 'bo')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        im.show()
        buf.close()
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
        test_image_return = np.zeros((100,100), np.uint8)#test remove before submission
        print("LS")
        return (test_image_return, test_image_return, test_image_return)

    def fit_line_robust(self, data_points, threshold):
        """ Fits a line to the given data points using robust estimators
        :param data_points: a list of data points
        :param threshold: a threshold value (if > threshold, imples outlier)
        :return: a tuple containing the followings:
                    * An image showing the line along with the data points
                    * The thresholded image
                    * A segmented image
        """
        test_image_return = np.zeros((100, 100), np.uint8)  # test remove before submission
        print("RO")
        return (test_image_return, test_image_return, test_image_return)

    def fit_gaussian(self, data_points, threshold):
        """ Fits the data points to a gaussian
        :param data_points: a list of data points
        :param threshold: a threshold value (if < threshold, imples outlier)
        :return: a tuple containing the followings:
                    * An image showing the gaussian along with the data points
                    * The thresholded image
                    * A segmented image
        """
        test_image_return = np.zeros((100, 100), np.uint8)  # test remove before submission
        print("GA")
        return (test_image_return, test_image_return, test_image_return)
