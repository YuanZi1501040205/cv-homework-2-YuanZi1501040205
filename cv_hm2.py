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


parser = argparse.ArgumentParser()
parser.add_argument("-i1", "--img1_path", dest="img1_path", help="Specify the path to the first image", required=True)
parser.add_argument("-i2", "--img2_path", dest="img2_path", help="Specify the path to the second image", required=True)
parser.add_argument("-t", "--threshold", dest="threshold", help="Specify the threshold value", required=True)


def main():
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

    # Fitting a line to the data using total least square
    line_fitting_tls, thresholded_tls, segmented_tls = fitting_object.fit_line_tls(data_points, args.threshold)
    output_path = output_dir + "line_fitting_tls_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_path, line_fitting_tls)
    output_path = output_dir + "thresholded_tls_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_path, thresholded_tls)
    output_path = output_dir + "segmented_tls_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_path, segmented_tls)

    # Fitting a line to the data using robust estimators
    line_fitting_ransac, thresholded_ransac, segmented_ransac = fitting_object.fit_line_tls(data_points, args.threshold)
    output_path = output_dir + "line_fitting_ransac_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_path, line_fitting_ransac)
    output_path = output_dir + "thresholded_ransac_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_path, thresholded_ransac)
    output_path = output_dir + "segmented_ransac_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_path, segmented_ransac)

    # Fitting a data to a gaussian
    gaussian_fitting, thresholded_gaussian, segmented_gaussian = fitting_object.fit_gussian(data_points)
    output_path = output_dir + "gaussian_fitting_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_path, gaussian_fitting)
    output_path = output_dir + "thresholded_gaussian_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_path, thresholded_gaussian)
    output_path = output_dir + "segmented_gaussian_" + str(args.threshold) + "_" + datetime.now().strftime("%m%d-%H%M%S") + ".jpg"
    cv2.imwrite(output_path, segmented_gaussian)


if __name__ == "__main__":
    main()










