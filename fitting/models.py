class FittingModels:

    def create_dataPoints(self, image1, image2):
        """ Creates a list of data points given two images
        :param image1: first image
        :param image2: second image
        :return: a list of data points
        """

    def plot_data(self, data_points):
        """ Plots the data points
        :param data_points:
        :return: an image
        """

    def fit_line_tls(self, data_points, threshold):
        """ Fits a line to the given data points using total least squares
        :param data_points: a list of data points
        :param threshold: a threshold value
        :return: a tuple containing the followings:
                    * An image showing the line along with the data points
                    * The thresholded image
                    * An segmented image
        """

    def fit_line_ransac(self, data_points, threshold):
        """ Fits a line to the given data points using robust estimators
        :param data_points: a list of data points
        :param threshold: a threshold value
        :return: a tuple containing the followings:
                    * An image showing the line along with the data points
                    * The thresholded image
                    * An segmented image
        """

    def fit_gussian(self, data_points, threshold):
        """ Fits the data points to a gaussian
        :param data_points: a list of data points
        :return: a tuple containing the followings:
                    * An image showing the gaussian along with the data points
                    * The thresholded image
                    * An segmented image
        """
