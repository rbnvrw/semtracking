import pandas

import numpy
import matplotlib.pyplot as plt

from semtracking import particlefinder, simulatedimage

errors = pandas.DataFrame(columns=['r', 'r_err', 'r_err_std', 'x_err', 'x_err_std', 'y_err', 'y_err_std'])
radii = numpy.linspace(5, 30, 25)
width = 1024
height = 943


class TestImage:
    def __init__(self, number, r):
        """
        Setup test image
        """
        self.number = number
        self.radius = r
        self.image = self.generate_image()

    def generate_image(self):
        """
        Generate the test image

        :rtype : semtracking.SimulatedImage
        :return:
        """
        image = simulatedimage.SimulatedImage(shape=(1024, 943), dtype=numpy.float, radius=self.radius, noise=0.3)
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
        frame = frame.sort_values(by=sort_columns)
        return frame.reset_index(drop=True)

    def get_coords_dataframe(self, add_r=False):
        """
        Get a dataframe with the generated coordinates
        :rtype : pandas.DataFrame
        """
        coords = self.image.coords
        coords_df = pandas.DataFrame(data=coords, columns=['y', 'x'])

        # Add r
        if add_r:
            coords_df['r'] = pandas.Series(self.radius, index=coords_df.index)

        coords_df = self.sort_dataframe(coords_df, ['x', 'y'])
        return coords_df


for r in radii:
    test = TestImage(height / r, r)
    generated_image = test.image()

    finder = particlefinder.ParticleFinder(generated_image)
    fits = finder.locate_particles(test.number)
    fits.drop(['dev'], axis=1, inplace=True)
    fits = test.sort_dataframe(fits, ['x', 'y'])

    coords_df = test.get_coords_dataframe(True)
    coords_df = test.sort_dataframe(coords_df, ['x', 'y'])

# Plot
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.set_xlabel('r')
ax1.set_ylabel('r_fit / r')
ax1.scatter(errors['r'], errors['r_err'])
ax1.errorbar(errors['r'], errors['r_err'], yerr=errors['r_err_std'])

ax2.set_xlabel('r')
ax2.set_ylabel('x_fit / x')
ax2.scatter(errors['r'], errors['x_err'])
ax2.errorbar(errors['r'], errors['x_err'], yerr=errors['x_err_std'])

ax3.set_xlabel('r')
ax3.set_ylabel('y_fit / y')
ax3.scatter(errors['r'], errors['y_err'])
ax3.errorbar(errors['r'], errors['y_err'], yerr=errors['y_err_std'])

plt.show()
