import unittest
import numpy as np
from numpy.testing import assert_allclose
from semtracking import simulatedimage as si
from semtracking import particlefinder as pf
from pandas import DataFrame, Series


class TestFindParticles(unittest.TestCase):
    def setUp(self):
        """
        Setup test image
        """
        self.number = 100
        self.radius = 15.0
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

    def test_locate_particles(self):
        """
        Test locating particles
        """
        generated_image = self.image()

        finder = pf.ParticleFinder(generated_image)
        fits = finder.locate_particles(self.number)

        fits.drop(['dev'], axis=1, inplace=True)

        fits = self.sort_dataframe(fits, ['x', 'y'])

        coords_df = self.get_coords_dataframe(True)
        coords_df = self.sort_dataframe(coords_df, ['x', 'y'])

        self.assert_allclose_sorted(fits['r'], coords_df['r'], rtol=0.1,
                                    err_msg='Locate particles: radius not accurate.')
        self.assert_allclose_sorted(fits['x'], coords_df['x'], rtol=0.1, err_msg='Locate particles: x not accurate.')
        self.assert_allclose_sorted(fits['y'], coords_df['y'], rtol=0.1, err_msg='Locate particles: y not accurate.')


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
