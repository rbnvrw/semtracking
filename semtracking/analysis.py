from skimage.feature import canny
from skimage.transform import hough_circle
from pandas import DataFrame
import numpy as np
from scipy.spatial import cKDTree


def find_hough_circle(im, sigma=2, low_threshold=10, high_threshold=50, r_range=(5, 20), n=200):
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


def eliminate_duplicates(f, separation, pos_columns, mass_column):
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
