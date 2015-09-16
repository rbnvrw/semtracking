import pims
from semtracking import util
from semtracking import analysis
from semtracking import report
from semtracking import plot

directory = r'C:\\Users\\verweij\\PycharmProjects\\testfiles\\'

for fn in util.gen_tif_paths(directory):
    im = pims.Bioformats(directory + fn[1] + '.tif')
    im = im[0][:-64]
    f = analysis.locate_hough_circles(im)
    plot.save_hough_circles(f, im, directory + '\\fits\\' + fn[1])
    #report.save_circles_to_csv(f, directory + fn[1])
