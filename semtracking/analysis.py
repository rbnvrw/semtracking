from numpy.linalg import LinAlgError
import skimage.feature
import skimage.filters
import skimage.util
from skimage import transform
import pandas
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline
import plot
import matplotlib

# Store fits so that we can use it in event callback
fits_for_user_check = pandas.DataFrame()


def find_hough_circles(image, sigma=2, r_range=(5, 40), n=200):
    """
    Locate circles using canny & hough transform
    :param sigma:
    :param r_range:
    :param n:
    :return:
    """
    # Find edges
    edges = skimage.feature.canny(image, sigma=sigma)

    # Find circles
    hough = transform.hough_circle(edges, np.linspace(*r_range, num=50))

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
    mask = ((f2['y'] - f2['r']) >= 0) & ((f2['x'] - f2['r']) >= 0) & (
        (f2['y'] + f2['r']) <= image.shape[0]) & (
               (f2['x'] + f2['r']) <= image.shape[1])

    f2 = f2[mask]

    return f2.reset_index(drop=True)


def get_intensity_interpolation(image, r, xc, yc, n, rad_range, spline_order=3):
    """
    Create a spline representation of the intensity
    :param r:
    :param xc:
    :param yc:
    :param n:
    :param rad_range:
    :param spline_order:
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
                    min(round(yc + r + rad_range[1] + 1), image.shape[0]))
    bound_x = slice(max(round(xc - r + rad_range[0]), 0),
                    min(round(xc + r + rad_range[1] + 1), image.shape[1]))

    interpolation = RectBivariateSpline(np.arange(bound_y.start, bound_y.stop),
                                        np.arange(bound_x.start, bound_x.stop),
                                        image[bound_y, bound_x], kx=spline_order,
                                        ky=spline_order, s=0)

    mean_intensity = np.mean(image)
    intensity = interpolation(y_rad, x_rad, grid=False)

    # check for points outside the image; set these to mean
    mask = ((y_rad >= bound_y.stop) | (y_rad < bound_y.start) |
            (x_rad >= bound_x.stop) | (x_rad < bound_x.start))
    intensity[mask] = mean_intensity

    return intensity, (x, y, step_x, step_y)


def refine_circle(image, r, yc, xc, spline_order=3):
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
    n = int(2 * np.pi * np.sqrt(r ** 2))

    rad_range = (-r, r)

    # Get intensity in spline representation
    intensity, (x, y, step_x, step_y) = get_intensity_interpolation(image, r, xc, yc, n, rad_range, spline_order)

    # mean intensity
    mean_intensity = np.mean(image)

    # First check if intensity is high enough for this to be a particle
    mean_particle_intensity = np.mean(intensity)

    if mean_particle_intensity < mean_intensity:
        return pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])

    # Get points with max negative slope
    max_neg_slopes = get_max_neg_slopes(intensity)

    # Calculate new circle coordinates
    coord_new = spline_coords_to_normal(max_neg_slopes, rad_range, x, y, step_x, step_y)

    # Fit a circle to the calculated coordinates
    fit = fit_circle(coord_new, r, yc, xc)

    return fit


def refine_circles(image, f, spline_order=3):
    """
    Make refinement to Hough circles by using the edge from light to dark
    :param f:
    :param spline_order:
    :return:
    """
    assert image.ndim == 2
    rs = f['r']
    ycs = f['y']
    xcs = f['x']

    fit = pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])
    for r, yc, xc in zip(rs, ycs, xcs):
        fit = pandas.concat(
            [fit, refine_circle(image, r, yc, xc, spline_order)], ignore_index=True)

    return fit


def locate_hough_circles(image, sigma=2, r_range=(5, 40), n=200):
    """
    Locate & refine circles using hough transform
    :param sigma:
    :param r_range:
    :param n:
    :return:
    """
    f = find_hough_circles(image, sigma, r_range, n)
    f = refine_circles(image, f)
    return f


def on_pick(event):
    """
    User clicked on a fit
    :param event:
    """
    global fits_for_user_check

    # Get index from label
    fit_number = int(event.artist.get_label())

    if not fits_for_user_check['remove'][fit_number]:
        event.artist.set_edgecolor('r')
        plot.set_annotation_color(fit_number, 'r')
        fits_for_user_check['remove'][fit_number] = True
    else:
        event.artist.set_edgecolor('b')
        plot.set_annotation_color(fit_number, 'b')
        fits_for_user_check['remove'][fit_number] = False

    event.canvas.draw()


def user_check_fits(filename, image, micron_per_pixel):
    """
    Let user manually check fits, removing them by clicking
    :return:
    """
    global fits_for_user_check
    fits_for_user_check = pandas.DataFrame.from_csv(filename)

    # Set scale in pixels
    fits_for_user_check /= micron_per_pixel

    fits_for_user_check['remove'] = False
    plot.plot_fits_for_user_confirmation(fits_for_user_check, image, on_pick)

    mask = (fits_for_user_check['remove'] == False)
    fits_for_user_check = fits_for_user_check[mask]

    fits_for_user_check.drop('remove', axis=1, inplace=True)

    # Update indices
    fits_for_user_check.index = range(1, len(fits_for_user_check) + 1)

    return fits_for_user_check


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


def get_max_neg_slopes(intensity):
    """
    Get positions with maximum negative slope
    :param intensity:
    :return:
    """
    # take differential to see changes in intensity
    intensity_diff = np.diff(intensity, 1)

    # give larger weight to spots in the center, as this is only a refinement of the hough circle
    threshold = np.min(intensity_diff) * 0.01
    w = intensity_diff.shape[1] / 2.0
    weights = [threshold * (w - abs(x)) if abs(x) < w / 2 else 0 for x in np.linspace(-w, w, intensity_diff.shape[1])]

    # create weighted intensity difference array
    weighted_intensity_diff = intensity_diff + weights

    # find the minima
    max_neg_slopes = np.argmin(weighted_intensity_diff, axis=1)

    # Get mean/std dev
    mean = np.mean(max_neg_slopes)
    dev = np.std(max_neg_slopes)

    # Remove outliers
    mask = np.abs(max_neg_slopes - mean) > dev
    max_neg_slopes[mask] = mean

    return max_neg_slopes
