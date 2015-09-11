from __future__ import (division, unicode_literals, print_function)

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from .algebraic import fitEllipseStraight
from scipy.interpolate import RectBivariateSpline
from .plot import annotate_ellipse_mpl
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
import pickle

FIND_CENTER_ATOL = 20.0
REFINE_RADIUS_RTOL = 0.5
REFINE_RADIUS_ATOL = 30.0
REFINE_CENTER_ATOL = 30.0
MIN_CONTOUR_LENGTH = 24

def find_ellipse(image):
    """ Thresholds the image, finds the longest contour and fits an ellipse
    to this contour.

    Parameters
    ----------
    image : 2D numpy array of numbers

    Returns
    -------
    yr, xr, yc, xc when dimension order was y, x (common)
    xr, yr, xc, yc when dimension order was x, y
    """
    assert image.ndim == 2
    thresh = threshold_otsu(image)
    binary = image > thresh
    contours = find_contours(binary, 0.5, fully_connected='high')
    if len(contours) == 0:
        raise ValueError('No contours found')
    
    # eliminate short contours
    contours = [c for c in contours if len(c) >= MIN_CONTOUR_LENGTH]

    # fit circles to the rest, keep the one with lowest residual deviation
    result = [np.nan] * 4
    residual = None
    for c in contours:
        try:
            xr, yr, xc, yc = fitEllipseStraight(c)
            x, y = c.T
            r = np.sum((((xc - x)/xr)**2 + ((yc - y)/yr)**2 - 1)**2)/len(c)
            if residual is None or r < residual:
                result = xr, yr, xc, yc
                residual = r
        except np.linalg.LinAlgError:
            pass

    return result


def find_ellipsoid(image3d):
    """ Fits ellipses to all three projections of the 3D image and returns
    center coordinates and priciple radii. The function is taylored for
    resonant scanning confocal images, which are blurred in x direction.

    Parameters
    ----------
    image3d : 3D numpy array of numbers

    Returns
    -------
    zr, yr, xr, zc, yc, xc
    """
    assert image3d.ndim == 3

    # Y, X projection, use y radius because resonant scanning in x direction.
    image = np.mean(image3d, axis=0)
    yr, xr, yc, xc = find_ellipse(image)

    # Z, X projection
    image = np.mean(image3d, axis=1)
    zr, xr2, zc, xc2 = find_ellipse(image)

    # Z, Y projection (noisy with resonant scanning)
    image = np.mean(image3d, axis=2)
    zr2, yr2, zc2, yc2 = find_ellipse(image)

    assert_allclose([xc, yc, zc],
                    [xc2, yc2, zc2], rtol=0, atol=FIND_CENTER_ATOL,
                    err_msg='Initial guess centers have inconsistent values.')

    return zr, yr, xr, zc, yc, xc


def refine_ellipse(image, params, n=None, rad_range=None, maxfit_size=2,
                   spline_order=3, threshold=0.1, show=False):
    """ Interpolates the image along lines perpendicular to the ellipse. 
    The maximum along each line is found using linear regression of the
    descrete derivative.

    Parameters
    ----------
    data : 2d numpy array of numbers
        Image indices are interpreted as (y, x)
    center: tuple of floats
    radius: tuple of floats
    n: integer
        number of points on the ellipse that are used for refine
    rad_range: tuple of floats
        length of the line (distance inwards, distance outwards)
    maxfit_size: integer
        pixels around maximum pixel that will be used in linear regression
    spline_order: integer
        interpolation order for edge crossections
    threshold: float
        a threshold is calculated based on the global maximum
        fitregions are rejected if their average value is lower than this

    Returns
    -------
    yr, xr, yc, xc

    """
    if not np.all([x > 0 for x in params]):
        raise ValueError("All yc, xc, yr, xr params should be positive")
    assert image.ndim == 2
    yr, xr, yc, xc = params
    if rad_range is None:
        rad_range = (-min(yr, xr) / 2, min(yr, xr) / 2)
    if n is None:
        n = int(2*np.pi*np.sqrt((yr**2 + xr**2) / 2))
    t = np.linspace(-np.pi, np.pi, n, endpoint=False)
    normalAngle = np.arctan2(xr*np.sin(t), yr*np.cos(t))
    x = xr * np.cos(t) + xc
    y = yr * np.sin(t) + yc
    step_x = np.cos(normalAngle)
    step_y = np.sin(normalAngle)
    steps = np.arange(rad_range[0], rad_range[1] + 1, 1)[np.newaxis, :]

    x_rad = x[:, np.newaxis] + steps * step_x[:, np.newaxis]
    y_rad = y[:, np.newaxis] + steps * step_y[:, np.newaxis]
    if show:
        for i in range(len(x)):
            plt.plot([x_rad[i, 0], x_rad[i, -1]], [y_rad[i, 0], y_rad[i, -1]],
                     color='y')

    # create a spline representation of the vesicle region
    bound_y = slice(max(round(yc - yr + rad_range[0]), 0),
                    min(round(yc + yr + rad_range[1] + 1), image.shape[0]))
    bound_x = slice(max(round(xc - xr + rad_range[0]), 0),
                    min(round(xc + xr + rad_range[1] + 1), image.shape[1]))
    interpl = RectBivariateSpline(np.arange(bound_y.start, bound_y.stop),
                                  np.arange(bound_x.start, bound_x.stop),
                                  image[bound_y, bound_x], kx=spline_order,
                                  ky=spline_order, s=0)
    intensity = interpl(y_rad, x_rad, grid=False)

    # check for points outside the image; set these to 0
    mask = ((y_rad >= bound_y.stop) | (y_rad < bound_y.start) |
            (x_rad >= bound_x.stop) | (x_rad < bound_x.start))
    intensity[mask] = 0

    # identify the regions around the max value
    maxes = np.argmax(intensity[:, maxfit_size:-maxfit_size], axis=1) + maxfit_size
    fitregion = np.array([[np.take(i, m + offs) for offs in range(-maxfit_size, maxfit_size + 1)] 
                          for (i, m) in zip(intensity, maxes)])
    
    # identify regions that are outside of the image
    threshold = threshold * fitregion.max()  # relative to global maximum
    in_image = (fitregion > 0).all(1) & (fitregion.mean(1) > threshold)

    # fit max using linear regression
    intdiff = np.diff(fitregion[in_image], 1)
    x_norm = np.arange(-maxfit_size + 0.5, maxfit_size + 0.5) # is normed because symmetric, x_mean = 0
    y_mean = np.mean(intdiff, axis=1, keepdims=True)
    y_norm = intdiff - y_mean
    slope = np.sum(x_norm[np.newaxis, :] * y_norm, 1) / np.sum(x_norm * x_norm)
    r_dev = - y_mean[:, 0] / slope
    mask = np.isfinite(r_dev) * (r_dev > -maxfit_size + 0.5) * (r_dev < maxfit_size - 0.5)

    # calculate new coords
    r_dev = r_dev + maxes[in_image] + rad_range[0]
    x_new = (x[in_image] + r_dev*step_x[in_image])[mask]
    y_new = (y[in_image] + r_dev*step_y[in_image])[mask]
    coord_new = np.vstack([y_new, x_new]).T
    
    # fit ellipse
    fit = fitEllipseStraight(coord_new)

    if show:
        plt.imshow(image, cmap=plt.get_cmap('gray'))
        plt.xlim(np.min(x_new) - rad_range[1], np.max(x_new) + rad_range[1])
        plt.ylim(np.min(y_new) - rad_range[1], np.max(y_new) + rad_range[1])
        plt.plot(x_new, y_new, marker='.')
        annotate_ellipse_mpl(fit, image, ax=plt.gca())

    return fit, coord_new


def refine_ellipsoid(image3d, p, n_xy=None, n_xz=None, rad_range=None,
                     maxfit_size=2, spline_order=3, threshold=0.1, show=False):
    """ Refines coordinates of a 3D ellipsoid, starting from given parameters.
    It assumes resonant images, it only analyzes YX and ZX projections.

    Parameters
    ----------
    image3d : 3D numpy array
    p: tuple of floats
        (zr, yr, xr, zc, yr, xr) coordinates of ellipsoid center
    n_xy: integer
        number of points on the ellipse that are used for refine in xy plane
    n_xz: integer
        number of points on the ellipse that are used for refine in xz plane
    rad_range: tuple of floats
        length of the line (distance inwards, distance outwards)
    maxfit_size: integer
        pixels around maximum pixel that will be used in linear regression
    spline_order: integer
        interpolation order for edge crossections
    threshold: float
        a threshold is calculated based on the global maximum
        fitregions are rejected if their average value is lower than this

    Returns
    -------
    (zr, yr, xr, zc, yc, xc), contour
    """
    assert image3d.ndim == 3
    zr0, yr0, xr0, zc0, yc0, xc0 = p
    # refine X, Y radius and center on XY middle
    middle_slice = image3d[int(zc0)] * (1 - zc0 % 1) + \
                   image3d[int(zc0) + 1] * (zc0 % 1)
    if show:
        plt.figure()
    (yr, xr, yc, xc), r = refine_ellipse(middle_slice, (yr0, xr0, yc0, xc0),
                                         n_xy, rad_range, maxfit_size,
                                         spline_order, threshold, show)

    # refine Z radius and center on ZX middle (not ZY, is blurred by resonant)
    middle_slice = image3d[:, int(yc0)] * (1 - yc0 % 1) + \
                   image3d[:, int(yc0) + 1] * (yc0 % 1)
    if show:
        plt.figure()
    (zr, _, zc, _), _ = refine_ellipse(middle_slice, (zr0, xr0, zc0, xc0),
                                       n_xz, rad_range, maxfit_size,
                                       spline_order, threshold, show)

    assert_allclose([xr, yr, zr],
                    [xr0, yr0, zr0], REFINE_RADIUS_RTOL, REFINE_RADIUS_ATOL,
                    err_msg='Refined value differs extremly from initial value.')
                    
    assert_allclose([xc, yc, zc],
                    [xc0, yc0, zc0], rtol=0, atol=REFINE_CENTER_ATOL,
                    err_msg='Refined value differs extremly from initial value.')

    return (zr, yr, xr, zc, yc, xc), r


def locate_ellipsoid(frame, n_xy=None, n_xz=None, rad_range=None,
                     maxfit_size=2, spline_order=3, threshold=0.1, show=False):
    """Locates an ellipsoid in a 3D image and returns center coordinates and
    priciple radii. The function is taylored for resonant scanning confocal
    images, which are blurred in x direction.

    Parameters
    ----------
    image3d: 3D ndarray
    n_xy: integer
        number of points on the ellipse that are used for refine in xy plane
    n_xz: integer
        number of points on the ellipse that are used for refine in xz plane
    rad_range: tuple of floats
        length of the line (distance inwards, distance outwards)
    maxfit_size: integer
        pixels around maximum pixel that will be used in linear regression
    spline_order: integer
        interpolation order for edge crossections
    threshold: float
        a threshold is calculated based on the global maximum
        fitregions are rejected if their average value is lower than this

    Returns
    -------
    Series with zr, yr, xr, zc, yc, xc indices, ndarray with (y, x) contour
    """
    assert frame.ndim == 3
    columns = ['zr', 'yr', 'xr', 'zc', 'yc', 'xc']
    try:
        params = find_ellipsoid(frame)
        params, r = refine_ellipsoid(frame, params, n_xy, n_xz, rad_range,
                                     maxfit_size, spline_order, threshold,
                                     show)
    except Exception:
        params = [np.nan] * 6
        r = None        

    return  pd.Series(params, index=columns), r


def batch_ellipsoid(frames, n_xy=None, n_xz=None, rad_range=None,
                    maxfit_size=2, spline_order=3, threshold=0.1,
                    contour_fn=None):
    """Locates ellipses an ellipsoid in a 3D image and returns
    center coordinates and priciple radii. The function is taylored for
    resonant scanning confocal images, which are blurred in x direction.
    Only accepts oblate or prolate ellipsoids, with xr = yr.

    Parameters
    ----------
    frames : iterable of images
    tol : number or tuple of numbers, Optional
        when given a value, checks disagreements of all 5 parameters
    refine: boolean
        determines wether the refine procedure will be run
    n_xy: integer
        number of points on the ellipse that are used for refine in xy plane
    n_xz: integer
        number of points on the ellipse that are used for refine in xz plane
    rad_range: tuple of floats
        range in which crossection fit is done, in pixels
    maxfit_size: integer
        pixels around max pixel that will be used in linear regression fit
    spline_order: integer
        interpolation order for edge crossections
    threshold: float
        minimum average intensity of fitregion (=pixels around max pixel)
        relative to max intensity of all fitregions
    contour_fn: filename
        Filename to save the contour.

    Returns
    -------
    DataFrame with zr, yr, xr, zc, yc, xc, frame columns
    """
    v = []
    contours = dict()
    for i, frame in enumerate(frames):
        if hasattr(frame, 'frame_no') and frame.frame_no is not None:
            frame_no = frame.frame_no
        else:
            frame_no = i
        vesicle, r = locate_ellipsoid(frame, n_xy, n_xz, rad_range,
                                      maxfit_size, spline_order, threshold)
        vesicle['frame'] = frame_no
        v.append(vesicle)
        if contour_fn is not None and r is not None:
            contours[frame_no] = r
    if contour_fn is not None:
        with open(contour_fn, 'wb') as contourdump:
            pickle.dump(contours, contourdump)

    result = pd.DataFrame(v)
    result['frame'] = result['frame'].astype(int)
    return result
