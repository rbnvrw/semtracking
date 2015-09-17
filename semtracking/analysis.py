from skimage.feature import canny
from skimage.transform import hough_circle
from pandas import DataFrame, concat
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline


def auto_canny(image, sigma=2):
    """
    Calculate upper and lower thresholds for canny
    :param image:
    :param sigma:
    :return:
    """

    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (2 / sigma) * v))
    upper = int(min(255, (4 * sigma) * v))
    edges = canny(image, sigma, lower, upper)

    # return the edged image
    return edges


def find_hough_circles(im, sigma=2, r_range=(5, 40), n=200):
    """
    Locate circles using canny & hough transform
    :param im:
    :param sigma:
    :param r_range:
    :param n:
    :return:
    """
    # Find edges
    edges = auto_canny(im, sigma)

    # Find circles
    hough = hough_circle(edges, np.arange(*r_range))

    # Find indices
    indices = hough.ravel().argsort()[-n:]
    indices = np.unravel_index(indices, hough.shape)

    # Find mass and radii
    f = DataFrame(np.array(indices).T, columns=['r', 'y', 'x'])
    f['r'] += r_range[0]
    f['mass'] = hough[indices]

    # Eliminate duplicates
    r = f.r.median()
    f2 = eliminate_duplicates(f, (r, r), ['y', 'x'], 'mass')

    # Eliminate circles outside the image
    mask = ((f2['y'] - f2['r']) >= 0) & ((f2['x'] - f2['r']) >= 0) & ((f2['y'] + f2['r']) <= im.shape[0]) & (
        (f2['x'] + f2['r']) <= im.shape[1])

    f2 = f2[mask]

    return f2.reset_index(drop=True)


def refine_circle(r, yc, xc, im, n=None, spline_order=3):
    """
    Make refinement to hough circles by using the edge from light to dark
    :param r:
    :param yc:
    :param xc:
    :param im:
    :param n:
    :param spline_order:
    :return:
    """
    if n is None:
        n = int(2 * np.pi * np.sqrt(r ** 2))

    rad_range = (-r, r)

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

    fit = fit_circle(coord_new, r, yc, xc)

    return fit


def refine_circles(f, im, n=None, spline_order=3):
    """
    Make refinement to hough circles by using the edge from light to dark
    :param f:
    :param im:
    :param n:
    :param spline_order:
    :return:
    """
    assert im.ndim == 2
    rs = f['r']
    ycs = f['y']
    xcs = f['x']

    fit = DataFrame(columns=['r', 'y', 'x', 'dev'])
    for r, yc, xc in zip(rs, ycs, xcs):
        fit = concat(
            [fit, refine_circle(r, yc, xc, im, n, spline_order)],
            ignore_index=True)

    return fit


def locate_hough_circles(im, sigma=2, r_range=(5, 40), n=200):
    """
    Locate & refine circles using hough transform
    :param im:
    :param sigma:
    :param r_range:
    :param n:
    :return:
    """
    f = find_hough_circles(im, sigma, r_range, n)
    f = refine_circles(f, im, n)
    return f


def eliminate_duplicates(f, separation, pos_columns, mass_column):
    """
    Eliminate duplicate circles
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
        duplicates = cKDTree(positions, 30).query_pairs(1.5)
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


def remove_outlier_points(features, r, yc, xc):
    """
    Remove outliers that are not on the circle
    :param features:
    :param r:
    :param yc:
    :param xc:
    :return:
    """
    x = features[:, 1]
    y = features[:, 0]

    mask = np.sqrt((x - xc) ** 2 + (y - yc) ** 2) <= 1.2 * r

    features = np.vstack([y, x]).T

    features = features[mask]

    return features


def fit_circle(features, r, yc, xc):
    """
     From x, y points, returns an algebraic fit of a circle
    (not optimal least squares, but very fast)
    :param features:
    :param r:
    :param yc:
    :param xc:
    :return: returns center, radius and rms deviation from fitted
    """
    # remove points that are clearly not on circle
    features = remove_outlier_points(features, r, yc, xc)

    # Get x,y
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
