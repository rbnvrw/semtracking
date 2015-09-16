from skimage.feature import canny
from skimage.transform import hough_circle
from pandas import DataFrame, concat
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline


def find_hough_circle(im, sigma=2, low_threshold=10, high_threshold=50, r_range=(5, 20), n=200):
    """

    :param im:
    :param sigma:
    :param low_threshold:
    :param high_threshold:
    :param r_range:
    :param n:
    :return:
    """
    edges = canny(im, sigma, low_threshold, high_threshold)
    hough = hough_circle(edges, np.arange(*r_range))
    indices = hough.ravel().argsort()[-n:]
    indices = np.unravel_index(indices, hough.shape)
    f = DataFrame(np.array(indices).T, columns=['r', 'y', 'x'])
    f['r'] += r_range[0]
    f['mass'] = hough[indices]
    r = f.r.median()
    f2 = eliminate_duplicates(f, (r, r), ['y', 'x'], 'mass')
    return f2.reset_index(drop=True)


def refine_hough_circle(r, yc, xc, im, n=None, rad_range=None, spline_order=3):
    if rad_range is None:
        rad_range = (-r, r)
    if n is None:
        n = int(2 * np.pi * np.sqrt(r ** 2))

    t = np.linspace(-np.pi, np.pi, n, endpoint=False)
    normalangle = np.arctan2(r * np.sin(t), r * np.cos(t))
    x = r * np.cos(t) + xc
    y = r * np.sin(t) + yc
    step_x = np.cos(normalangle)
    step_y = np.sin(normalangle)
    steps = np.arange(rad_range[0], rad_range[1] + 1, 1)[np.newaxis, :]

    x_rad = x[:, np.newaxis] + steps * step_x[:, np.newaxis]
    y_rad = y[:, np.newaxis] + steps * step_y[:, np.newaxis]

    # create a spline representation of the colloid region
    bound_y = slice(max(round(yc - r + rad_range[0]), 0),
                    min(round(yc + r + rad_range[1] + 1), im.shape[0]))
    bound_x = slice(max(round(xc - r + rad_range[0]), 0),
                    min(round(xc + r + rad_range[1] + 1), im.shape[1]))

    interpl = RectBivariateSpline(np.arange(bound_y.start, bound_y.stop),
                                  np.arange(bound_x.start, bound_x.stop),
                                  im[bound_y, bound_x], kx=spline_order,
                                  ky=spline_order, s=0)

    intensity = interpl(y_rad, x_rad, grid=False)

    # check for points outside the image; set these to 0
    mask = ((y_rad >= bound_y.stop) | (y_rad < bound_y.start) |
            (x_rad >= bound_x.stop) | (x_rad < bound_x.start))
    intensity[mask] = 0

    # First check if intensity is high enough for this to be a particle
    mean_particle_intensity = np.mean(intensity)
    mean_intensity = np.mean(im)

    if mean_particle_intensity < mean_intensity:
        return DataFrame(columns=['r', 'y', 'x', 'dev'])

    # identify the regions around the max negative slope
    intdiff = np.diff(intensity, 1)
    max_neg_slopes = np.argmin(intdiff, axis=1)

    # calculate new coords
    r_dev = max_neg_slopes + rad_range[0] + 0.5
    x_new = (x + r_dev * step_x)
    y_new = (y + r_dev * step_y)
    coord_new = np.vstack([y_new, x_new]).T

    fit = fit_circle(coord_new)

    return fit


def refine_hough_circles(f, im, n=None, rad_range=None, spline_order=3):
    """

    :param f:
    :param im:
    """

    assert im.ndim == 2
    rs = f['r']
    ycs = f['y']
    xcs = f['x']

    fit = DataFrame(columns=['r', 'y', 'x', 'dev'])
    for r, yc, xc in zip(rs, ycs, xcs):
        fit = concat(
            [fit, refine_hough_circle(r, yc, xc, im, n, rad_range, spline_order)],
            ignore_index=True)

    return fit


def locate_hough_circles(im, sigma=2, low_threshold=10, high_threshold=50, r_range=(5, 40), n=200):
    """

    :param im:
    :param sigma:
    :param low_threshold:
    :param high_threshold:
    :param r_range:
    :param n:
    :return:
    """
    f = find_hough_circle(im, sigma, low_threshold, high_threshold, r_range, n)
    f = refine_hough_circles(f, im, n)
    return f


def eliminate_duplicates(f, separation, pos_columns, mass_column):
    """

    :param f:
    :param separation:
    :param pos_columns:
    :param mass_column:
    :return:
    """
    result = f.copy()
    while True:
        # Rescale positions, so that pairs are identified below a distance
        # of 1. Do so every iteration (room for improvement?)
        positions = result[pos_columns].values / list(separation)
        mass = result[mass_column].values
        duplicates = cKDTree(positions, 30).query_pairs(1)
        if len(duplicates) == 0:
            break
        to_drop = []
        for pair in duplicates:
            # Drop the dimmer one.
            if np.equal(*mass.take(pair, 0)):
                # Rare corner case: a tie!
                # Break ties by sorting by sum of coordinates, to avoid
                # any randomness resulting from cKDTree returning a set.
                dimmer = np.argsort(np.sum(positions.take(pair, 0), 1))[0]
            else:
                dimmer = np.argmin(mass.take(pair, 0))
            to_drop.append(pair[dimmer])
        result.drop(to_drop, inplace=True)
    return result


def fit_circle(features):
    # from x, y points, returns an algebraic fit of a circle
    # (not optimal least squares, but very fast)
    # returns center, radius and rms deviation from fitted

    x = features[:, 1]
    y = features[:, 0]

    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center in reduced coordinates (uc, vc):
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv = np.sum(u * v)
    Suu = np.sum(u ** 2)
    Svv = np.sum(v ** 2)
    Suuv = np.sum(u ** 2 * v)
    Suvv = np.sum(u * v ** 2)
    Suuu = np.sum(u ** 3)
    Svvv = np.sum(v ** 3)

    # Solving the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    try:
        uc, vc = np.linalg.solve(A, B)
    except:
        return DataFrame(columns=['r', 'y', 'x', 'dev'])

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calculation of all distances from the center (xc_1, yc_1)
    Ri_1 = np.sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
    R_1 = np.mean(Ri_1)
    sqrdeviation = np.mean((Ri_1 - R_1) ** 2)

    data = {'r': [R_1], 'y': [yc_1], 'x': [xc_1], 'dev': [sqrdeviation]}
    fit = DataFrame(data)

    return fit
