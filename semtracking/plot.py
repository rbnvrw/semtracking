from os import path, makedirs, sep
import matplotlib.pyplot as plt


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
    for i in f.index:
        plt.gca().add_patch(plt.Circle((f.loc[i].x, f.loc[i].y), radius=f.loc[i].r, fc='None', ec='b', ls='solid'))
    plt.show()
    return plt.gca()


def plot_circle_on_image(f, im):
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
    for i in f.index:
        plt.gca().add_patch(
            plt.Circle((f.loc[i].x, f.loc[i].y), radius=f.loc[i].r, fc='None', ec='b', ls='solid', lw=0.3, label=i))
    plt.show()
    return plt.gca()


def plot_circle(f):
    """
    Make a plot of the image and the found circles
    :param f: Dataframe with x, y, r
    :param im: Image
    :return: The axis
    """
    for i in f.index:
        plt.gca().add_patch(
            plt.Circle((f.loc[i].x, f.loc[i].y), radius=f.loc[i].r, fc='None', ec='b', ls='solid', lw=0.3, label=i))
    plt.show()
    return plt.gca()


def plot_fits_for_user_confirmation(f, im, pick_callback):
    """
    Ask user to check if all fits are correct.
    Clicking on a fit removes it from the results.
    :param f:
    :param im:
    :param pick_callback:
    """
    _imshow_style = dict(origin='lower', interpolation='none',
                         cmap=plt.cm.gray)
    plt.clf()
    plt.cla()
    plt.imshow(im, **_imshow_style)
    for i in f.index:
        circle = plt.Circle((f.loc[i].x, f.loc[i].y), radius=f.loc[i].r, fc='None', ec='b', ls='solid', lw=0.3,
                            label=i)

        # Enable picking
        circle.set_picker(10)

        plt.gca().add_patch(circle)

        plt.gca().annotate(i, (f.loc[i].x, f.loc[i].y), color='b', weight='normal',
                           size=8, ha='center', va='center')

    plt.gca().set_title('Please check the result. Click on a circle to toggle removal. Close to confirm.')
    plt.gcf().canvas.mpl_connect('pick_event', pick_callback)
    plt.show()


def set_annotation_color(index, color):
    """
    Remove particle index from the current plot
    :param index:
    """
    children = plt.gca().get_children()
    children = [c for c in children if isinstance(c, plt.Annotation) and int(c._text) == index]

    for child in children:
        child.set_color(color)


def save_fits(f, im, filename, dpi=300, linewidth=0.3):
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
                         cmap=plt.cm.gray)
    plt.clf()
    plt.cla()
    plt.imshow(im, **_imshow_style)
    for i in f.index:
        circle = plt.Circle((f.loc[i].x, f.loc[i].y), radius=f.loc[i].r, fc='None', ec='b', ls='solid', lw=linewidth,
                            label=i)
        plt.gca().add_patch(circle)

        plt.gca().annotate(i, (f.loc[i].x, f.loc[i].y), color='w', weight='normal',
                           fontsize=3, ha='center', va='center')
    plt.savefig(path.abspath(path.normpath(directory + sep + filename)) + '_fit.tif', dpi=dpi)
    plt.close()


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
    ax = plot_image(im)
    ax.scatter(x, y, c='r', s=20)


def plot_scatter_and_fit(x, y, fit, im):
    """
    Overlay a scatter plot + fit on image
    :param x:
    :param y:
    :param im:
    """
    ax = plot_image(im)
    ax.scatter(x, y, c='r', s=20)
    plot_circle(fit)
