import matplotlib.pyplot as plt


def plot_hough_circle(f, im):
    """

    :param f:
    :param im:
    :return:
    """
    plt.clf()
    plt.cla()
    _imshow_style = dict(origin='lower', interpolation='none',
                         cmap=plt.cm.gray)

    plt.imshow(im, **_imshow_style)
    plt.gca().invert_yaxis()
    for i in f.index:
        plt.gca().add_patch(plt.Circle((f.loc[i].x, f.loc[i].y), radius=f.loc[i].r, fc='None', ec='b', ls='solid'))
    return plt.gca()


def save_hough_circles(f, im, filename):
    """

    :param f:
    :param im:
    :return:
    """
    plt.clf()
    plt.cla()
    _imshow_style = dict(origin='lower', interpolation='none',
                         cmap=plt.cm.gray)

    plt.imshow(im, **_imshow_style)
    plt.gca().invert_yaxis()
    for i in f.index:
        plt.gca().add_patch(plt.Circle((f.loc[i].x, f.loc[i].y), radius=f.loc[i].r, fc='None', ec='b', ls='solid'))
    plt.savefig(filename + '_fit.tif')


def plot_image(im):
    _imshow_style = dict(origin='lower', interpolation='none',
                         cmap=plt.cm.gray)

    plt.imshow(im, **_imshow_style)
    plt.show()
    return plt.gca()


def plot_scatter(x, y, im):
    plot_image(im)
    plt.scatter(x, y, c='r')
