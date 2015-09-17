import pims
import os
from semtracking import util
from semtracking import analysis
from semtracking import report
from semtracking import plot

directory = os.path.abspath(os.path.normpath("c:/Users/verweij/PycharmProjects/testfiles/"))

for filename in util.gen_img_paths(directory):
    path = os.path.join(directory, filename)
    im = pims.Bioformats(path + '.tif')
    micron_per_pixel = im.calibration
    im = im[0][:-64]
    f = analysis.locate_hough_circles(im)
    plot.save_hough_circles(f, im, path)
    report.save_circles_to_csv(f, path, micron_per_pixel)
    break
