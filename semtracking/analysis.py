from numpy.linalg import LinAlgError
import skimage.feature
import skimage.filters
from skimage import transform
import pandas
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline


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
    edges = skimage.feature.canny(im, sigma=sigma)

    # Find circles
    hough = transform.hough_circle(edges, np.arange(*r_range))

    # Find indices
    indices = hough.ravel().argsort()[-n:]
    indices = np.unravel_index(indices, hough.shape)

    # Find mass and radii
    f = pandas.DataFrame(np.array(indices).T, columns=['r', 'y', 'x'])
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


def get_intensity_interpolation(r, xc, yc, n, rad_range, spline_order, im):
    """
    Create a spline representation of the intensity
    :param r:
    :param xc:
    :param yc:
    :param n:
    :param rad_range:
    :param spline_order:
    :param im:
    :return:
    """
    t = np.linspace(-np.pi, np.pi, n, endpoint=False)
    normal_angle = np.arctan2(r * np.sin(t), r * np.cos(t))
    x = r * np.cos(t) + xc
    y = r * np.sin(t) + yc
    step_x = np.cos(normal_angle)
    step_y = np.sin(normal_angle)
    steps = np.arange(rad_range[0], rad_range[1] + 1, 1)[np.newaxis, :]

    x_rad = x[:, np.newaxis] + steps * step_x[:, np.newaxis]
    y_rad = y[:, np.newaxis] + steps * step_y[:, np.newaxis]

    # create a spline representation of the colloid region
    bound_y = slice(max(round(yc - r + rad_range[0]), 0),
                    min(round(yc + r + rad_range[1] + 1), im.shape[0]))
    bound_x = slice(max(round(xc - r + rad_range[0]), 0),
                    min(round(xc + r + rad_range[1] + 1), im.shape[1]))

    interpolation = RectBivariateSpline(np.arange(bound_y.start, bound_y.stop),
                                        np.arange(bound_x.start, bound_x.stop),
                                        im[bound_y, bound_x], kx=spline_order,
                                        ky=spline_order, s=0)

    mean_intensity = np.mean(im)
    intensity = interpolation(y_rad, x_rad, grid=False)

    # check for points outside the image; set these to mean
    mask = ((y_rad >= bound_y.stop) | (y_rad < bound_y.start) |
            (x_rad >= bound_x.stop) | (x_rad < bound_x.start))
    intensity[mask] = mean_intensity

    return intensity, (x, y, step_x, step_y)


def get_max_neg_slopes(intensity, r):
    """
    Get positions with maximum negative slope
    :param intensity:
    :param r:
    :return:
    """
    # identify the regions around the max negative slope
    # use sobel filter for edge detection and smoothening the image
    intensity = skimage.filters.sobel(intensity)

    # take differential to see changes in intensity
    intensity_diff = np.diff(intensity, 1)

    # give larger weight to spots in the center, as this is only a refinement of the hough circle
    threshold = np.min(intensity_diff) * 0.4
    distance_from_center = [threshold * i for j in
                            (range(0, intensity_diff.shape[1] / 2), range(intensity_diff.shape[1] / 2, 0, -1))
                            for i in j]

    # create weighted intensity difference array
    weighted_intensity_diff = intensity_diff + distance_from_center

    # find the minima
    max_neg_slopes = np.argmin(weighted_intensity_diff, axis=1)

    # find the intensity values at these positions
    max_neg_slope_vals = np.take(intensity_diff, max_neg_slopes, axis=1)

    # Remove values under intensity diff threshold
    threshold = np.min(max_neg_slope_vals) * 0.4
    threshold_mask = (max_neg_slope_vals > threshold)
    max_neg_slope_vals[threshold_mask] = 0

    # rescale positions
    max_neg_slopes -= 0.5 * np.ones(max_neg_slopes.shape)

    return max_neg_slopes


def spline_coords_to_normal(max_neg_slopes, rad_range, x, y, step_x, step_y):
    """
    Calculate normal coordinates from spline representation coordinates
    :param max_neg_slopes:
    :param rad_range:
    :param x:
    :param y:
    :param step_x:
    :param step_y:
    :return:
    """
    r_dev = max_neg_slopes + rad_range[0] + 0.5
    x_new = (x + r_dev * step_x)
    y_new = (y + r_dev * step_y)
    coord_new = np.vstack([y_new, x_new]).T

    return coord_new


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

    # Get intensity in spline representation
    intensity, (x, y, step_x, step_y) = get_intensity_interpolation(r, xc, yc, n, rad_range, spline_order, im)

    # mean intensity
    mean_intensity = np.mean(im)

    # First check if intensity is high enough for this to be a particle
    mean_particle_intensity = np.mean(intensity)

    if mean_particle_intensity < mean_intensity:
        return pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])

    # Get points with max negative slope
    max_neg_slopes = get_max_neg_slopes(intensity, r)

    # Calculate new circle coordinates
    coord_new = spline_coords_to_normal(max_neg_slopes, rad_range, x, y, step_x, step_y)

    # Fit a circle to the calculated coordinates
    fit = fit_circle(coord_new, r, yc, xc)

    return fit


def refine_circles(f, im, n=None, spline_order=3):
    """
    Make refinement to Hough circles by using the edge from light to dark
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

    fit = pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])
    for r, yc, xc in zip(rs, ycs, xcs):
        fit = pandas.concat(
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

    # Remove points farther than 1.2 r from the center
    mask = np.sqrt((x - xc) ** 2 + (y - yc) ** 2) <= 1.2 * r

    features = np.vstack([y, x]).T

    # Apply the mask
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
    s_uv = np.sum(u * v)
    s_uu = np.sum(u ** 2)
    s_vv = np.sum(v ** 2)
    s_uuv = np.sum(u ** 2 * v)
    s_uvv = np.sum(u * v ** 2)
    s_uuu = np.sum(u ** 3)
    s_vvv = np.sum(v ** 3)

    # Solving the linear system
    a = np.array([[s_uu, s_uv], [s_uv, s_vv]])
    b = np.array([s_uuu + s_uvv, s_vvv + s_uuv]) / 2.0
    try:
        solution, _, _, _ = np.linalg.lstsq(a, b)
    except LinAlgError:
        return pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])

    # Calculate new centers
    uc = solution[0]
    vc = solution[1]

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calculation of all distances from the center (xc_1, yc_1)
    ri_1 = np.sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
    r_1 = np.mean(ri_1)
    square_deviation = np.mean((ri_1 - r_1) ** 2)

    data = {'r': [r_1], 'y': [yc_1], 'x': [xc_1], 'dev': [square_deviation]}
    fit = pandas.DataFrame(data)

    return fit
