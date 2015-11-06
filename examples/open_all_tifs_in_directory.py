import pims
import os
import numpy as np
from semtracking import util
from semtracking import particlefinder as pf
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

    # Calculate size range
    min_size = max(0.5/micron_per_pixel, 3)
    max_size = min(1.5/micron_per_pixel, im.frame_shape[0]*0.5)

    im = im[0][:-64]
    im = np.flipud(im)

    # Locate and refine circles
    finder = pf.ParticleFinder(im)
    f = finder.locate_particles(size_range=(min_size, max_size))

    # Save fit images
    plot.save_fits(f, im, path)

    # Generate data files and save
    report.save_circles_to_csv(f, path, micron_per_pixel)
    report.save_circles_to_csv_grouped(f, path, micron_per_pixel)
