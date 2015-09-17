import matplotlib.pyplot as plt
import os


def plot_hough_circle(f, im):
    """
    Make a plot of the image and the found circles
    :param f: Dataframe with x, y, r
    :param im: Image
    :return: The axis
    """
    plt.clf()
    plt.cla()
    _imshow_style = dict(origin='lower', interpolation='none',
                         cmap=plt.cm.gray)

    plt.imshow(im, **_imshow_style)
    plt.gca().invert_yaxis()
    for i in f.index:
        plt.gca().add_patch(plt.Circle((f.loc[i].x, f.loc[i].y), radius=f.loc[i].r, fc='None', ec='b', ls='solid'))
    plt.show()
    return plt.gca()


def save_hough_circles(f, im, filename, dpi=300, linewidth=0.3):
    """
    Save plot of image and Hough circles to a file in a subdirectory
    :param f: Dataframe with x, y, r
    :param im: Image
    :param filename: The filename
    :param dpi: DPI of saved image
    :param linewidth: Linewidth of the circles
    :return:
    """
    directory = os.path.abspath(os.path.normpath(os.path.dirname(filename) + os.sep + 'fits'))

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.basename(filename)
    _imshow_style = dict(origin='lower', interpolation='none',
                         cmap=plt.cm.gray)
    plt.clf()
    plt.cla()
    plt.imshow(im, **_imshow_style)
    plt.gca().invert_yaxis()
    for i in f.index:
        circle = plt.Circle((f.loc[i].x, f.loc[i].y), radius=f.loc[i].r, fc='None', ec='b', ls='solid', lw=linewidth,
                            label=i)
        plt.gca().add_patch(circle)

        plt.gca().annotate(i, (f.loc[i].x, f.loc[i].y), color='w', weight='normal',
                           fontsize=3, ha='center', va='center')
    plt.savefig(os.path.abspath(os.path.normpath(directory + os.sep + filename)) + '_fit.tif', dpi=dpi)


def plot_image(im):
    """
    Plots an image
    :param im:
    :return:
    """
    _imshow_style = dict(origin='lower', interpolation='none',
                         cmap=plt.cm.gray)

    plt.imshow(im, **_imshow_style)
    plt.show()
    return plt.gca()


def plot_scatter(x, y, im):
    """
    Overlay a scatter plot on image
    :param x:
    :param y:
    :param im:
    """
    plot_image(im)
    plt.scatter(x, y, c='r')
