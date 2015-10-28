import unittest
import numpy as np

from numpy.testing import assert_allclose
from semtracking import simulatedimage as si
from semtracking import analysis as an
from semtracking import plot
from pandas import DataFrame


class TestFindParticles(unittest.TestCase):
    def setUp(self):
        self.number = 100
        self.radius = 15
        self.image = self.generate_image()

    def generate_image(self):
        image = si.SimulatedImage(shape=(1024, 943), dtype=np.float, radius=self.radius, noise=0.3)
        image.draw_features(self.number, separation=3 * self.radius)
        return image

    def sort_dataframe(self, frame, sort_columns):
        return frame.sort(sort_columns)

    def test_guess_average_radius(self):
        generated_image = self.image()

        radius = an.guess_average_radius(generated_image)

        self.assertAlmostEqual(self.radius, radius, places=0,
                               msg='Guessed radius not accurate: guessed %d, actual %d' % (radius, self.radius))

    def test_find_hough_circles(self):
        generated_image = self.image()

        result = an.find_hough_circles(generated_image, sigma=2, r_range=(self.radius / 3, self.radius * 3),
                                       n=self.number * 2)

        result.drop(['r', 'mass'], axis=1, inplace=True)
        result = self.sort_dataframe(result, ['x', 'y'])

        coords = self.image.coords
        coords_df = DataFrame(data=coords, columns=['y', 'x'])
        coords_df = self.sort_dataframe(result, ['x', 'y'])

        assert_allclose(result, coords_df, rtol=1e-3, err_msg='Hough transform not finding all particles.')


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
