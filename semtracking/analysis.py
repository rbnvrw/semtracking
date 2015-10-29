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
from skimage.morphology import disk, closing
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.measure import regionprops

# Store fits so that we can use it in event callback
fits_for_user_check = pandas.DataFrame()


def locate_circular_particles(image, sigma=2, n=200):
    """
    Locate & refine circles using hough transform
    :param sigma:
    :param r_range:
    :param n:
    :return:
    """
    image = normalize_image(image)
    guessed_r = guess_average_radius(image)
    r_range = (guessed_r / 3, guessed_r * 3)
    f = find_hough_circles(image, sigma, r_range, n)
    f = refine_circles(image, f)
    return f


def normalize_image(image):
    """
    Normalize image
    :param image:
    :return:
    """
    image = image.astype(np.float64)
    abs_max = np.max(np.abs(image))
    return image / abs_max


def guess_average_radius(image):
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, disk(3))

    # remove artifacts connected to image border
    cleared = bw.copy()
    clear_border(cleared)

    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1

    radii = [r.equivalent_diameter / 2.0 for r in regionprops(label_image)]
    average_r = np.mean(radii)

    return average_r


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

    # Eliminate duplicates, but first keep touching ones
    r = f.r.max()
    f = eliminate_duplicates(f, 0.5 * r, ['y', 'x'], 'mass')

    # Eliminate circles outside the image
    mask = ((f['y'] - f['r']) >= 0) & ((f['x'] - f['r']) >= 0) & (
        (f['y'] + f['r']) <= image.shape[0]) & (
               (f['x'] + f['r']) <= image.shape[1])

    f = f[mask]

    return f.reset_index(drop=True)


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
    n = int(np.round(2 * np.pi * np.sqrt(r ** 2)))

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
    max_slopes = get_max_slopes(intensity)

    if len(max_slopes) == 0:
        return pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])

    # Calculate new circle coordinates
    coord_new = spline_coords_to_normal(max_slopes, rad_range, x, y, step_x, step_y)

    # Fit a circle to the calculated coordinates
    fit = fit_circle(coord_new).reset_index()

    return fit


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


def flatten_multi_columns(col):
    """

    :param col:
    :param sep:
    :return:
    """
    if not type(col) is tuple:
        return col
    else:
        return col[0]


def merge_points_same_index(data):
    """

    :param data:
    :return:
    """
    data.index = np.round(data.index)
    grouped = data.groupby(by=data.index)

    merged_data = grouped.aggregate([np.mean])
    merged_data.columns = merged_data.columns.map(flatten_multi_columns)

    return merged_data


def get_max_slopes(intensity):
    """
    Get positions with maximum negative slope
    :param intensity:
    :return:
    """
    # Find edges
    intensity = normalize_image(intensity)
    gx, gy = np.gradient(intensity)

    # Find local maxima
    local_maxes = skimage.feature.peak_local_max(gx, min_distance=1, threshold_rel=0.4, exclude_border=True)

    if len(local_maxes) == 0:
        return []

    # Detect subpixel corners
    subpix_corners = skimage.feature.corner_subpix(intensity, local_maxes)

    # Create dataframe of x values, indexing by y and take mean for points with same y
    local_maxes_df = pandas.DataFrame(data={'x': subpix_corners[:, 1], 'y': subpix_corners[:, 0]})

    # Try to interpolate missing x values
    local_maxes_df = local_maxes_df.interpolate(method='nearest', axis=0).ffill().bfill()

    # Set the index
    local_maxes_df = local_maxes_df.set_index('y', drop=False, verify_integrity=False)

    # Merge points with the same y value
    local_maxes_df = merge_points_same_index(local_maxes_df)

    # Generate index of all y values of intensity array
    index = np.arange(0, intensity.shape[0], 1)

    # Reindex with all y values, filling with NaN's
    local_maxes_df = local_maxes_df.reindex(index, fill_value=np.nan)

    # Try to interpolate missing x values
    local_maxes_df = local_maxes_df.interpolate(method='nearest', axis=0).ffill().bfill()

    # Remove outlier x's
    mean_x = np.mean(local_maxes_df['x'])
    mask = np.abs(local_maxes_df['x'] - mean_x) > mean_x*0.3
    local_maxes_df[mask] = mean_x

    return list(local_maxes_df['x'])


def spline_coords_to_normal(max_slopes, rad_range, x, y, step_x, step_y):
    """
    Calculate normal coordinates from spline representation coordinates
    :param max_slopes:
    :param rad_range:
    :param x:
    :param y:
    :param step_x:
    :param step_y:
    :return:
    """
    r_dev = max_slopes - abs(rad_range[0])
    x_new = (x + r_dev * step_x)
    y_new = (y + r_dev * step_y)
    coord_new = np.vstack([x_new, y_new]).T

    return coord_new


def fit_circle(features):
    """
     From x, y points, returns an algebraic fit of a circle
    (not optimal least squares, but very fast)
    :param features:
    :param r:
    :param yc:
    :param xc:
    :return: returns center, radius and rms deviation from fitted
    """
    # Get x,y
    x = features[:, 0]
    y = features[:, 1]

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

    # Check if dev is not too high
    if square_deviation / r_1 > 0.1:
        return pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])

    data = {'r': [r_1], 'y': [yc_1], 'x': [xc_1], 'dev': [square_deviation]}
    fit = pandas.DataFrame(data)

    return fit


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


def eliminate_duplicates(f, separation, pos_columns, mass_column):
    """
    Eliminate duplicate circles
    :param f:
    :param separation:
    :param pos_columns:
    :param mass_column:
    :return:
    """
    result = f.drop_duplicates().reset_index(drop=True)
    while True:
        # Rescale positions, so that pairs are identified below a distance
        # of 1. Do so every iteration (room for improvement?)
        positions = result[pos_columns].values
        mass = result[mass_column].values
        duplicates = cKDTree(positions, 30).query_pairs(3 * separation)
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
