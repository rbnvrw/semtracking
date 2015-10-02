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

    # Locate and refine circles
    circle_finder = analysis.CircleFinder(im)
    circle_finder.locate_hough_circles()

    # Show result for manual checking
    f = circle_finder.user_check_fits()

    # Save fit images
    plot.save_hough_circles(f, im, path)

    # Generate data files and save
    report.save_circles_to_csv(f, path, micron_per_pixel)
