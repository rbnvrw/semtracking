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
        frame = frame.sort(sort_columns)
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
    path = os.path.join(directory, 'plot.pdf')
    errors = pandas.DataFrame(columns=['r', 'num_diff', 'r_diff', 'x_diff', 'y_diff', 'r_diff_std', 'x_diff_std', 'y_diff_std'])
    radii = numpy.arange(5, 30, 1)
    width = 1024
    height = 943
    runs = 10

    for r in radii:
        for run in numpy.arange(1, runs, 1):
            # Change number so each run is "random"
            num = int(min([height / r, 100])+run)

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

            if abs(num_diff) > 0:
                plot.save_fits(fits, generated_image, os.path.join(directory, 'mismatch_'+str(r)+'_'+str(run)+'.tiff'))

            errors = errors.append({
                'r': r,
                'num_diff': num_diff,
                'r_diff': numpy.mean(r_diff),
                'x_diff': numpy.mean(x_diff),
                'y_diff': numpy.mean(y_diff),
                'r_diff_std': numpy.std(r_diff),
                'x_diff_std': numpy.std(x_diff),
                'y_diff_std': numpy.std(y_diff)
            }, ignore_index=True)

    # Plot
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.set_xlabel('r')
    ax1.set_ylabel('(n_r - n_f) / n_r')
    ax1.scatter(errors['r'], errors['num_diff'])

    ax2.set_xlabel('r')
    ax2.set_ylabel('(r_r - r_f) / r_r')
    ax2.errorbar(errors['r'], errors['r_diff'], yerr=errors['r_diff_std'], fmt='o')

    ax3.set_xlabel('r')
    ax3.set_ylabel('(x_r - x_f) / x_r')
    ax3.errorbar(errors['r'], errors['x_diff'], yerr=errors['x_diff_std'], fmt='o')

    ax4.set_xlabel('r')
    ax4.set_ylabel('(y_r - y_f) / y_r')
    ax4.errorbar(errors['r'], errors['y_diff'], yerr=errors['y_diff_std'], fmt='o')

    f.suptitle('Errors in fitting, ' + str(runs) + ' runs')

    plt.savefig(path)


if __name__ == "__main__":
    main(sys.argv[1:])
