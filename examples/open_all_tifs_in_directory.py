import pims
from semtracking import util
from semtracking import analysis
from semtracking import report
from semtracking import plot

directory = r'C:\\Users\\verweij\\PycharmProjects\\testfiles\\'

for filename in util.gen_img_paths(directory):
    im = pims.Bioformats(directory + filename + '.tif')
    micron_per_pixel = im.calibration
    im = im[0][:-64]
    f = analysis.locate_hough_circles(im)
    plot.save_hough_circles(f, im, directory + '\\fits\\' + filename)
    report.save_circles_to_csv(f, directory + filename, micron_per_pixel)
