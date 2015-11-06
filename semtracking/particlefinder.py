import skimage.feature, skimage.transform
import scipy.interpolate, scipy.ndimage, scipy.spatial
import numpy as np
import pandas
import plot


class ParticleFinder:
    local_max_min_distance = 5

    def __init__(self, image):
        """
        Class for finding circular particles
        :param image:
        """
        self.image = image
        self.n = 100
        self.size_range = (5, 30)
        self.mean = np.mean(self.image)
        self.min = np.min(self.image)
        self.max = np.max(self.image)

    def locate_particles(self, n=100, size_range=(5, 30)):
        """
        Find circular particles in the image
        :rtype : pandas.DataFrame
        :param n:
        :return:
        """
        self.n = n
        self.size_range = size_range
        # 1. Detect blobs in image
        blobs = self.locate_circles()

        if blobs.empty:
            return pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])

        # 2. Find circles
        fit = pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])
        for i, blob in blobs.iterrows():
            fit = pandas.concat([fit, self.find_circle(blob)], ignore_index=True)

        return fit

    def locate_circles(self):
        """
        Locate blobs in the image by using a Laplacian of Gaussian method
        :rtype : pandas.DataFrame
        :return:
        """
        num_r = min(abs(self.size_range[1] - self.size_range[0]), 30)
        radii = np.linspace(self.size_range[0], self.size_range[1], num_r)

        # Find edges
        edges = skimage.feature.canny(self.image)

        circles = skimage.transform.hough_circle(edges, radii)

        fit = pandas.DataFrame(columns=['r', 'y', 'x', 'accum'])
        for radius, h in zip(radii, circles):
            peaks = skimage.feature.peak_local_max(h, threshold_rel=0.5, num_peaks=self.n)
            accumulator = h[peaks[:, 0], peaks[:, 1]]
            fit = pandas.concat(
                [fit, pandas.DataFrame(data={'r': [radius] * peaks.shape[0], 'y': peaks[:, 0], 'x': peaks[:, 1],
                                             'accum': accumulator})], ignore_index=True)

        fit = self.merge_hough_same_values(fit)

        return fit

    @staticmethod
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

    @classmethod
    def merge_hough_same_values(self, data):
        """

        :param data:
        :return:
        """
        data = data.reset_index()
        data['x_div'] = np.round(data['x'] / 10.0)
        data['y_div'] = np.round(data['y'] / 10.0)

        grouped = data.groupby(by=['x_div', 'y_div']).aggregate(lambda x: tuple(x))
        grouped = grouped.drop(['index'], axis=1).reset_index()
        for col in ['x', 'y', 'r', 'accum']:
            grouped[col] = grouped.apply(lambda row: row[col][np.argmax(row['accum'])], axis=1)
        grouped.columns = grouped.columns.map(self.flatten_multi_columns)
        grouped = grouped.drop(['x_div', 'y_div'], axis=1).reset_index()

        while True:
            # Rescale positions, so that pairs are identified below a distance
            # of 1. Do so every iteration (room for improvement?)
            positions = grouped[['x', 'y']].values
            mass = grouped['accum'].values
            duplicates = scipy.spatial.cKDTree(positions, 30).query_pairs(np.max(grouped['r']), p=2.0, eps=0.1)
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
            grouped.drop(to_drop, inplace=True)

        return grouped

    def find_circle(self, blob):
        """
        Find a circle based on the blob
        :rtype : pandas.DataFrame
        :param blob:
        :return:
        """

        # Get intensity in spline representation
        rad_range = (-blob.r, blob.r)
        intensity, (x, y, step_x, step_y) = self.get_intensity_interpolation(blob, rad_range)

        if not self.check_intensity_interpolation(intensity):
            return pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])

        # Find the coordinates of the edge
        edge_coords = self.find_edge(intensity)

        if np.isnan(edge_coords.x).any():
            return pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])

        # Convert to cartesian
        coords = self.spline_coords_to_cartesian(edge_coords, rad_range, x, y, step_x, step_y)

        # Fit the circle
        fit = self.fit_circle(coords)

        return fit

    def get_intensity_interpolation(self, blob, rad_range):
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
        n = int(np.round(2 * np.pi * np.sqrt(blob.r ** 2)))
        spline_order = 3
        t = np.linspace(-np.pi, np.pi, n, endpoint=False)
        normal_angle = np.arctan2(blob.r * np.sin(t), blob.r * np.cos(t))
        x = blob.r * np.cos(t) + blob.x
        y = blob.r * np.sin(t) + blob.y
        step_x = np.cos(normal_angle)
        step_y = np.sin(normal_angle)
        steps = np.arange(rad_range[0], rad_range[1] + 1, 1)[np.newaxis, :]

        x_rad = x[:, np.newaxis] + steps * step_x[:, np.newaxis]
        y_rad = y[:, np.newaxis] + steps * step_y[:, np.newaxis]

        # create a spline representation of the colloid region
        bound_y = slice(max(round(blob.y - blob.r + rad_range[0]), 0),
                        min(round(blob.y + blob.r + rad_range[1] + 1), self.image.shape[0]))
        bound_x = slice(max(round(blob.x - blob.r + rad_range[0]), 0),
                        min(round(blob.x + blob.r + rad_range[1] + 1), self.image.shape[1]))

        interpolation = scipy.interpolate.RectBivariateSpline(np.arange(bound_y.start, bound_y.stop),
                                                              np.arange(bound_x.start, bound_x.stop),
                                                              self.image[bound_y, bound_x], kx=spline_order,
                                                              ky=spline_order, s=0)

        intensity = interpolation(y_rad, x_rad, grid=False)

        # check for points outside the image; set these to mean
        mask = ((y_rad >= bound_y.stop) | (y_rad < bound_y.start) |
                (x_rad >= bound_x.stop) | (x_rad < bound_x.start))
        intensity[mask] = self.mean

        return intensity, (x, y, step_x, step_y)

    @staticmethod
    def check_intensity_interpolation(intensity):
        """
        Check whether the intensity interpolation is bright on left, dark on right
        :rtype : bool
        :param intensity:
        :return:
        """
        sum_first_column = np.sum(intensity[:, 0])
        sum_last_column = np.sum(intensity[:, -1])
        return sum_first_column > sum_last_column

    @staticmethod
    def find_edge(intensity):
        """
        Find the edge of the particle
        :rtype : pandas.DataFrame
        :param intensity:
        :return:
        """
        # Divide image in two parts
        left = intensity[:, 0:intensity.shape[1] / 2]

        # Calculate means
        mean_left = np.mean(left)
        std_left = np.std(left)

        # Create binary mask
        mask = intensity >= mean_left - 2 * std_left
        # Fill holes in binary mask
        mask = scipy.ndimage.morphology.binary_fill_holes(mask)

        coords = pandas.DataFrame(columns=['x', 'y'])
        for y, row in enumerate(mask):
            indices_left = [i for i, col in enumerate(row) if col]
            indices_right = [i for i, col in enumerate(row) if not col]

            if len(indices_left) == 0 or len(indices_right) == 0:
                average_x = np.nan
            else:
                average_x = (indices_left[-1] + indices_right[0]) / 2.0

            new_coords = pandas.DataFrame(data={'x': [average_x], 'y': [y]})
            coords = pandas.concat([coords, new_coords], ignore_index=True)

        coords.fillna(np.nanmean(coords.x), inplace=True)

        return coords

    @staticmethod
    def spline_coords_to_cartesian(edge_coords, rad_range, x, y, step_x, step_y):
        """
        Calculate cartesian coordinates from spline representation coordinates
        :param max_slopes:
        :param rad_range:
        :param x:
        :param y:
        :param step_x:
        :param step_y:
        :return:
        """
        r_dev = edge_coords.x - abs(rad_range[0])
        x_new = (x + r_dev * step_x)
        y_new = (y + r_dev * step_y)
        coord_new = np.vstack([x_new, y_new]).T

        return coord_new

    @staticmethod
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
        except np.linalg.LinAlgError:
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
