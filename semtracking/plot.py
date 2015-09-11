import matplotlib.pyplot as plt


def plot_hough_circle(f, im):
    """

    :param f:
    :param im:
    :return:
    """
    _imshow_style = dict(origin='lower', interpolation='none',
                         cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(im, **_imshow_style)
    plt.gca().invert_yaxis()
    for i in f.index:
        plt.gca().add_patch(plt.Circle((f.loc[i].x, f.loc[i].y), radius=f.loc[i].r, fc='None', ec='b', ls='solid'))
    return plt.gca()
