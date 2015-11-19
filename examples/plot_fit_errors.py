import os

import pandas
import numpy
import matplotlib.pyplot as plt
import sys

from semtracking import particlefinder, simulatedimage, plot, util


class TestImage:
    def __init__(self, number, r, noise=0.3, shape=(1024, 943)):
        """
        Setup test image
        :param number:
        :param r:
        :param noise:
        :param shape:
        """
        self.number = number
        self.radius = r
        self.image = self.generate_image(noise, shape)

    def generate_image(self, noise=0.3, shape=(1024, 943)):
        """
        Generate the test image

        :param noise:
        :param shape:
        :rtype : semtracking.SimulatedImage
        :return:
        """
        image = simulatedimage.SimulatedImage(shape=shape, dtype=numpy.float, radius=self.radius, noise=noise)
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


def main(argv):
    """

    :param argv:
    """
    directory = util.get_directory_from_command_line(argv, os.path.basename(__file__))
    path = os.path.join(directory, 'error_plots', 'plot.png')
    errors = pandas.DataFrame(columns=['r', 'num_diff', 'r_diff', 'x_diff', 'y_diff'])
    radii = numpy.arange(5, 30, 1)
    width = 1024
    height = 943
    runs = 10

    for r in radii:
        num = int(min([height / r, 100]))
        for run in numpy.arange(1, runs, 1):
            test = TestImage(num, r, noise=0.3, shape=(width, height))
            generated_image = test.image()

            finder = particlefinder.ParticleFinder(generated_image)
            fits = finder.locate_particles(n=test.number * 1.5, size_range=(
                int(numpy.ceil(0.8 * min(radii))), int(numpy.ceil(1.2 * max(radii)))))
            fits = test.sort_dataframe(fits, ['x', 'y'])

            coords_df = test.get_coords_dataframe(True)
            coords_df = test.sort_dataframe(coords_df, ['x', 'y'])

            num_diff = (len(coords_df['r']) - len(fits['r'])) / len(coords_df['r'])
            r_diff = (coords_df['r'] - fits['r']) / coords_df['r']
            x_diff = (coords_df['x'] - fits['x']) / coords_df['x']
            y_diff = (coords_df['y'] - fits['y']) / coords_df['y']

            errors = errors.append({
                'r': r,
                'num_diff': num_diff,
                'r_diff': r_diff,
                'x_diff': x_diff,
                'y_diff': y_diff
            }, ignore_index=True)

    # Plot
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.set_xlabel('r')
    ax1.set_ylabel('(n_r - n_f) / n_r')
    ax1.scatter(errors['r'], errors['num_diff'])

    ax2.set_xlabel('r')
    ax2.set_ylabel('(r_r - r_f) / r_r')
    ax2.scatter(errors['r'], errors['r_diff'])

    ax3.set_xlabel('r')
    ax3.set_ylabel('(x_r - x_f) / x_r')
    ax3.scatter(errors['r'], errors['x_diff'])

    ax4.set_xlabel('r')
    ax4.set_ylabel('(y_r - y_f) / y_r')
    ax4.scatter(errors['r'], errors['y_diff'])

    plt.savefig(path)


if __name__ == "__main__":
    main(sys.argv[1:])
