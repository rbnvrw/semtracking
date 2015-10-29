import unittest
import numpy as np

from numpy.testing import assert_allclose
from semtracking import simulatedimage as si
from semtracking import analysis as an
from pandas import DataFrame, Series


class TestFindParticles(unittest.TestCase):
    def setUp(self):
        """
        Setup test image
        """
        self.number = 100
        self.radius = 15
        self.image = self.generate_image()

    def generate_image(self):
        """
        Generate the test image

        :rtype : semtracking.SimulatedImage
        :return:
        """
        image = si.SimulatedImage(shape=(1024, 943), dtype=np.float, radius=self.radius, noise=0.3)
        image.draw_features(self.number, separation=3 * self.radius)
        return image

    def sort_dataframe(self, frame, sort_columns):
        """
        Sort the dataframe by columns

        :rtype : pandas.DataFrame
        :param frame:
        :param sort_columns:
        :return:
        """
        frame = frame.reindex_axis(sorted(frame.columns), axis=1)
        frame = frame.reset_index(drop=True)
        frame = frame.sort(sort_columns)
        return frame.reset_index(drop=True)

    def get_coords_dataframe(self, add_r=False):
        """
        Get a dataframe with the generated coordinates
        :rtype : pandas.DataFrame
        """
        coords = self.image.coords
        coords_df = DataFrame(data=coords, columns=['y', 'x'])

        # Add r
        if add_r:
            coords_df['r'] = Series(self.radius, index=coords_df.index)

        coords_df = self.sort_dataframe(coords_df, ['x', 'y'])
        return coords_df

    def assert_allclose_sorted(self, a, b, rtol=0.1, err_msg="Assertion failed"):
        """

        :param a:
        :param b:
        :param rtol:
        :param err_msg:
        """
        assert_allclose(np.sort(a), np.sort(b), rtol=rtol, err_msg=err_msg)

    def test_guess_average_radius(self):
        """
        Test guessing of average radius
        """
        generated_image = self.image()

        radius = an.guess_average_radius(generated_image)

        self.assertAlmostEqual(self.radius, radius, places=0,
                               msg='Guessed radius not accurate: guessed %d, actual %d' % (radius, self.radius))

    def test_find_hough_circles(self):
        """
        Test finding of circles by Hough transform
        """
        generated_image = self.image()

        result = an.find_hough_circles(generated_image, sigma=2, r_range=(self.radius / 3, self.radius * 3),
                                       n=self.number * 2)

        result.drop(['r', 'mass'], axis=1, inplace=True)
        result = self.sort_dataframe(result, ['x', 'y'])

        coords_df = self.get_coords_dataframe()

        self.assert_allclose_sorted(result['x'], coords_df['x'], rtol=0.1, err_msg='Hough transform: x not accurate.')
        self.assert_allclose_sorted(result['y'], coords_df['y'], rtol=0.1, err_msg='Hough transform: y not accurate.')

    def test_refine_circles(self):
        """
        Test circle refinement function
        """
        generated_image = self.image()

        result = an.find_hough_circles(generated_image, sigma=2, r_range=(self.radius / 3, self.radius * 3),
                                       n=self.number * 2)
        fits = an.refine_circles(generated_image, result)
        fits.drop(['dev', 'index'], axis=1, inplace=True)

        fits = self.sort_dataframe(fits, ['x', 'y'])

        coords_df = self.get_coords_dataframe(True)
        coords_df = self.sort_dataframe(coords_df, ['x', 'y'])

        self.assert_allclose_sorted(fits['r'], coords_df['r'], rtol=0.1,
                                    err_msg='Circle refinement: radius not accurate.')
        self.assert_allclose_sorted(fits['x'], coords_df['x'], rtol=0.1, err_msg='Circle refinement: x not accurate.')
        self.assert_allclose_sorted(fits['y'], coords_df['y'], rtol=0.1, err_msg='Circle refinement: y not accurate.')


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
