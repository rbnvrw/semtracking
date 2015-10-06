import pims
import os
from semtracking import util
from semtracking import analysis
from semtracking import report
from semtracking import plot

# Replace with the target directory
directory = os.path.abspath(os.path.normpath("c:/Users/verweij/PycharmProjects/testfiles/"))

# Open all .tif's in the directory
for filename in util.gen_img_paths(directory):
    path = os.path.join(directory, filename)

    # Open with Bioformats
    im = pims.Bioformats(path + '.tif')

    # Set scale
    micron_per_pixel = im.calibration

    im = im[0][:-64]

    # Report filename
    report_file = os.path.abspath(
        os.path.normpath(directory + os.path.sep + 'report' + os.path.sep + filename + '_frame.csv'))
    if not os.path.isfile(report_file):
        continue

    f = analysis.user_check_fits(report_file, im, micron_per_pixel)

    # Remove old csv files
    summary_file = os.path.abspath(
        os.path.normpath(directory + os.path.sep + 'report' + os.path.sep + filename + '_summary.csv'))
    os.remove(summary_file)
    os.remove(report_file)

    # Remove old fit .tif file
    fits_directory = os.path.abspath(os.path.normpath(directory + os.path.sep + 'fits'))
    fit_file = os.path.abspath(os.path.normpath(fits_directory + os.path.sep + filename)) + '_fit.tif'
    os.remove(fit_file)

    # Save fit images
    plot.save_hough_circles(f, im, path)

    # Generate data files and save
    report.save_circles_to_csv(f, path, micron_per_pixel)
