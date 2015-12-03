import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from semtracking import simulatedimage as si
from semtracking import particlefinder as pf
from pandas import DataFrame, Series
from semtracking import plot

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
        image = si.SimulatedImage(shape=(1024, 943), radius=self.radius, noise=0.3)
        image.draw_features(self.number, separation=3 * self.radius)
        return image

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

        return coords_df

    @staticmethod
    def find_closest_index(expected_row, actual_df, intersection, column1, column2, tolerance=5.0):
        diffs = {i: abs(expected_row[column1] - actual_df.loc[i][column1]) + abs(
            expected_row[column2] - actual_df.loc[i][column2]) for i in intersection}

        key = min(diffs, key=diffs.get)

        if diffs[key] / 2.0 < tolerance:
            return key
        else:
            return False

    def find_intersections(self, expected_row, actual_df, closest_dict, column1, column2, tolerance=5.0):
        intersection = list(set(closest_dict[column1]).intersection(closest_dict[column2]))

        if len(intersection) == 0:
            return False
        else:
            return self.find_closest_index(expected_row, actual_df, intersection, column1, column2, tolerance)

    def find_closest_row_index(self, expected_row, actual_df, column1, column2, tolerance=5.0):
        closest = {c: (actual_df[c] - expected_row[c]).abs().argsort()[:10] for c in [column1, column2]}
        return self.find_intersections(expected_row, actual_df, closest, column1, column2, tolerance)

    def check_frames_difference(self, actual, expected, sizecol='r'):
        """
        Compare DataFrame items by index and column and
        raise AssertionError if any item is not equal.

        Ordering is unimportant, items are compared only by label.
        NaN and infinite values are supported.

        Parameters
        ----------
        actual : pandas.DataFrame
        expected : pandas.DataFrame
        use_close : bool, optional
            If True, use numpy.testing.assert_allclose instead of
            numpy.testing.assert_equal.

        """
        unmatched = 0
        value_diff = DataFrame(columns=expected.columns)

        for i, exp_row in expected.iterrows():
            # tolerance in pixels
            tolerance = max(exp_row[sizecol] * 0.1, 5.0)
            act_row_index = self.find_closest_row_index(exp_row, actual, 'x', 'y', tolerance)
            if act_row_index is False:
                unmatched += 1
                continue

            act_row = actual.loc[act_row_index]
            diff = exp_row - act_row[expected.columns]
            value_diff = value_diff.append(diff, ignore_index=True)
        return unmatched, value_diff

    def test_locate_particles(self):
        """
        Test locating particles
        """
        generated_image = self.image()

        finder = pf.ParticleFinder(generated_image)
        fits = finder.locate_particles(self.number)

        fits.drop(['dev'], axis=1, inplace=True)

        coords_df = self.get_coords_dataframe(True)

        unmatched, value_diff = self.check_frames_difference(fits, coords_df)

        assert_allclose(value_diff['r'], np.zeros_like(value_diff['r']), atol=0.03*self.radius, err_msg='Radius not accurate.')
        assert_allclose(value_diff['x'], np.zeros_like(value_diff['x']), atol=1, err_msg='X not accurate.')
        assert_allclose(value_diff['y'], np.zeros_like(value_diff['y']), atol=1, err_msg='Y not accurate.')
        assert_equal(unmatched, 0, 'Not all particles found, %d unmatched' % unmatched)

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
