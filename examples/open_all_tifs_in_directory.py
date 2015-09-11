import pims
from semtracking import util
from semtracking import analysis
from semtracking import plot
import matplotlib.pyplot as plt

directory = r'C:\\Users\\verweij\\PycharmProjects\\testfiles\\'

for fn in util.gen_tif_paths(directory):
    im = pims.Bioformats(directory + fn[1] + '.tif')
    im = im[0][:-64]
    break

f = analysis.locate_hough_circles(im)
plot.plot_hough_circle(f, im)
plt.show()
