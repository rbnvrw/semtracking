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

    r_range = (0.2/micron_per_pixel, 3/micron_per_pixel)

    im = im[0][:-64]

    # Locate and refine circles
    f = analysis.locate_hough_circles(im, r_range=r_range)

    # Save fit images
    plot.save_hough_circles(f, im, path)

    # Generate data files and save
    report.save_circles_to_csv(f, path, micron_per_pixel)
