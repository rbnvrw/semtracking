import skimage.feature
import simulatedimage as si
import scipy.interpolate
import numpy as np
import pandas
import plot


class CircularParticleFinder:
    local_max_min_distance = 5

    def __init__(self, image):
        self.image = image
        self.n = 100
        self.mean = np.mean(self.image)
        self.min = np.min(self.image)
        self.max = np.max(self.image)

    def locate_particles(self, n=100):
        self.n = n

        # 1. Detect blobs in image
        blobs = self.locate_blobs()

        # 2. Find circles
        fit = pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])
        for i, blob in blobs.iterrows():
            fit = pandas.concat([fit, self.find_circle(blob)], ignore_index=True)

    def locate_blobs(self):
        blobs = skimage.feature.blob_doh(self.image)
        blobs_df = pandas.DataFrame(columns=('y', 'x', 'r'), data=blobs)
        return blobs_df

    def find_circle(self, blob):
        # Get intensity in spline representation
        intensity, (x, y, step_x, step_y) = self.get_intensity_interpolation(blob)
        return pandas.DataFrame(columns=['r', 'y', 'x', 'dev'])

    def get_intensity_interpolation(self, blob):
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
        rad_range = (-blob.r, blob.r)
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


image = si.SimulatedImage(shape=(1024, 943), dtype=np.float, radius=30, noise=0.3)
image.draw_features(10, separation=3 * 30)
finder = CircularParticleFinder(image())
finder.locate_particles()
