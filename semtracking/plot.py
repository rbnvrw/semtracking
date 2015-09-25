from os import path, makedirs, sep

from matplotlib.pyplot import Circle, imshow, gca, savefig, scatter, show, cla, cm, clf


def plot_hough_circle(f, im):
    """
    Make a plot of the image and the found circles
    :param f: Dataframe with x, y, r
    :param im: Image
    :return: The axis
    """
    clf()
    cla()
    _imshow_style = dict(origin='lower', interpolation='none',
                         cmap=cm.gray)

    imshow(im, **_imshow_style)
    gca().invert_yaxis()
    for i in f.index:
        gca().add_patch(Circle((f.loc[i].x, f.loc[i].y), radius=f.loc[i].r, fc='None', ec='b', ls='solid'))
    show()
    return gca()


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
    directory = path.abspath(path.normpath(path.dirname(filename) + sep + 'fits'))

    if not path.exists(directory):
        makedirs(directory)

    filename = path.basename(filename)
    _imshow_style = dict(origin='lower', interpolation='none',
                         cmap=cm.gray)
    clf()
    cla()
    imshow(im, **_imshow_style)
    gca().invert_yaxis()
    for i in f.index:
        circle = Circle((f.loc[i].x, f.loc[i].y), radius=f.loc[i].r, fc='None', ec='b', ls='solid', lw=linewidth,
                        label=i)
        gca().add_patch(circle)

        gca().annotate(i, (f.loc[i].x, f.loc[i].y), color='w', weight='normal',
                       fontsize=3, ha='center', va='center')
    savefig(path.abspath(path.normpath(directory + sep + filename)) + '_fit.tif', dpi=dpi)


def plot_image(im):
    """
    Plots an image
    :param im:
    :return:
    """
    _imshow_style = dict(origin='lower', interpolation='none',
                         cmap=cm.gray)

    imshow(im, **_imshow_style)
    show()
    return gca()


def plot_scatter(x, y, im):
    """
    Overlay a scatter plot on image
    :param x:
    :param y:
    :param im:
    """
    ax = plot_image(im)
    ax.scatter(x, y, c='r', s=20)
