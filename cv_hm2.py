"""cv_hw1.py: Starter file to run howework 2"""

#Example Usage: ./cv_hw2 -i1 image1 -i2 image2 -t threshold
#Example Usage: python cv_hw1 -i image -k clusters -m rgb

import cv2
import sys
import argparse
from fitting.models import FittingModels
from datetime import datetime


def display_image(window_name, image):
    """A function to display image
    Note: only for debugging do not use for submission, Circle CI will fail"""
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)





def main():
    models = ['TL', 'RO', 'GA']

    parser = argparse.ArgumentParser()
    parser.add_argument("-i1", "--img1_path", dest="img1_path", help="Specify the path to the first image",
                        required=True)
    parser.add_argument("-i2", "--img2_path", dest="img2_path", help="Specify the path to the second image",
                        required=True)
    parser.add_argument("-m", "--model", dest="model",
                        help="Specify the fitting model (TL - total least squares, RO - robust fitting, GA - Gaussian Model",
                        required=True, type=str, choices = ['TL', 'RO', 'GA'])
    parser.add_argument("-t", "--threshold", dest="threshold", help="Specify the threshold value, TL and RO - any value, [0,1] - GA", required=True, type=float)


    args = parser.parse_args()

    image1 = cv2.imread(args.img1_path)

    if image1 is None:
        print("The path to the first image is incorrect\n")
        sys.exit()

    image2 = cv2.imread(args.img2_path)

    if image2 is None:
        print("The path to the second image is incorrect\n")
        sys.exit()


    
    fitting_object = FittingModels()

    # Creating data points
    data_points = fitting_object.create_dataPoints(image1, image2)

    output_dir = 'output/'

    # Plotting the data points
    data = fitting_object.plot_data(data_points)
    output_path = output_dir + "plotted_data" + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_path, data)

    if args.model == 'TL':
        # Fitting a line to the data using total least square
        line_fitting_tls, thresholded_tls, segmented_tls = fitting_object.fit_line_tls(data_points, args.threshold)
        output_path = output_dir + "line_fitting_tls_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
        cv2.imwrite(output_path, line_fitting_tls)
        output_path = output_dir + "thresholded_tls_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
        cv2.imwrite(output_path, thresholded_tls)
        output_path = output_dir + "segmented_tls_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
        cv2.imwrite(output_path, segmented_tls)

    if args.model == 'RO':
        # Fitting a line to the data using robust estimators
        line_fitting_robust, thresholded_robust, segmented_robust = fitting_object.fit_line_robust(data_points, args.threshold)
        output_path = output_dir + "line_fitting_robust_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
        cv2.imwrite(output_path, line_fitting_robust)
        output_path = output_dir + "thresholded_robust_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
        cv2.imwrite(output_path, thresholded_robust)
        output_path = output_dir + "segmented_robust_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
        cv2.imwrite(output_path, segmented_robust)

    if args.model == 'GA':
        # Fitting a data to a gaussian
        gaussian_fitting, thresholded_gaussian, segmented_gaussian = fitting_object.fit_gaussian(data_points, args.threshold)
        output_path = output_dir + "gaussian_fitting_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
        cv2.imwrite(output_path, gaussian_fitting)
        output_path = output_dir + "thresholded_gaussian_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
        cv2.imwrite(output_path, thresholded_gaussian)
        output_path = output_dir + "segmented_gaussian_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
        cv2.imwrite(output_path, segmented_gaussian)


if __name__ == "__main__":
    main()










