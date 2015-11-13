import pims
import os
import numpy as np
from semtracking import util
from semtracking import usercheckfits
from semtracking import report
from semtracking import plot
import re

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
    im = np.flipud(im)

    # Report filename
    report_file = os.path.abspath(
        os.path.normpath(directory + os.path.sep + 'report' + os.path.sep + filename + '_frame.csv'))
    if not os.path.isfile(report_file):
        continue

    checker = usercheckfits.UserCheckFits(report_file, micron_per_pixel)
    f = checker.user_check_fits(im)

    # Remove old csv files
    summary_file = os.path.abspath(
        os.path.normpath(directory + os.path.sep + 'report' + os.path.sep + filename + '_summary.csv'))
    if os.path.isfile(summary_file):
        os.remove(summary_file)
    os.remove(report_file)

    # Paths
    file_path_grouped = os.path.abspath(
        os.path.normpath(directory + os.path.sep + 'report' + os.path.sep + re.sub("_\d+$", "", filename)))
    report_file_grouped = file_path_grouped + '_grouped_report.csv'
    summary_file_grouped = file_path_grouped + '_grouped_summary.csv'
    if os.path.isfile(report_file_grouped):
        os.remove(report_file_grouped)
    if os.path.isfile(summary_file_grouped):
        os.remove(summary_file_grouped)

    # Remove old fit .tif file
    fits_directory = os.path.abspath(os.path.normpath(directory + os.path.sep + 'fits'))
    fit_file = os.path.abspath(os.path.normpath(fits_directory + os.path.sep + filename)) + '_fit.tif'
    if os.path.isfile(fit_file):
        os.remove(fit_file)

    # Save fit images
    plot.save_fits(f, im, path)

    # Generate data files and save
    report.save_circles_to_csv(f, path, micron_per_pixel)
    report.save_circles_to_csv_grouped(f, path, micron_per_pixel)
