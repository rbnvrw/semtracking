import os

import pandas
import numpy
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl

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
        image = simulatedimage.SimulatedImage(shape=shape, radius=self.radius, noise=noise)
        image.draw_features(self.number, separation=3 * self.radius)
        return image

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
        value_diff = pandas.DataFrame(columns=expected.columns)

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


def set_latex_params():
    fig_width_pt = 483.69687  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (numpy.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean + fig_width / 7  # height in inches
    fig_size = [fig_width, fig_height]

    plt.rcParams['text.latex.preamble'] = [
        r"\usepackage{mathpazo} \usepackage[separate-uncertainty=true]{siunitx} \DeclareSIUnit\Molar{\textsc{m}}"]

    pdf_with_latex = {  # setup matplotlib to use latex for output
                        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
                        "text.usetex": True,  # use LaTeX to write all text
                        "font.family": "serif",
                        "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
                        "font.sans-serif": [],
                        "font.monospace": [],
                        "axes.labelsize": 10,  # LaTeX default is 10pt font.
                        "font.size": 12,
                        "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
                        "xtick.labelsize": 10,
                        "ytick.labelsize": 10,
                        "figure.figsize": fig_size,
                        'axes.linewidth': .5,
                        'lines.linewidth': .5,
                        'patch.linewidth': .5,
                        'text.latex.unicode': True,
                        }
    mpl.rcParams.update(pdf_with_latex)


def main(argv):
    """

    :param argv:
    """
    directory = util.get_directory_from_command_line(argv, os.path.basename(__file__))
    path = os.path.join(directory, 'plot.pdf')
    filepath = os.path.join(directory, 'errors.csv')
    errors = pandas.DataFrame(
        columns=['r', 'num_diff', 'r_diff', 'x_diff', 'y_diff', 'r_diff_std', 'x_diff_std', 'y_diff_std'])
    radii = numpy.arange(5.0, 30.0, 1.0)
    width = 1024
    height = 943
    runs = 50

    set_latex_params()

    for index, r in enumerate(radii):
        for run in numpy.arange(1, runs, 1):
            num = round(min([height / r, 100]))

            test = TestImage(num, r, noise=0.3, shape=(width, height))
            generated_image = test.image()

            finder = particlefinder.ParticleFinder(generated_image)
            fits = finder.locate_particles(n=test.number * 1.5, size_range=(
                int(numpy.ceil(0.8 * min(radii))), int(numpy.ceil(1.2 * max(radii)))))

            coords_df = test.get_coords_dataframe(True)

            unmatched, value_diff = test.check_frames_difference(fits, expected=coords_df)

            errors = errors.append({
                'r': r,
                'num_diff': (len(coords_df['r']) - unmatched) / len(coords_df['r']),
                'r_diff': numpy.mean(value_diff['r']) / r,
                'x_diff': numpy.mean(value_diff['x']),
                'y_diff': numpy.mean(value_diff['y']),
                'r_diff_std': numpy.std(value_diff['r']) / r,
                'x_diff_std': numpy.std(value_diff['x']),
                'y_diff_std': numpy.std(value_diff['y'])
            }, ignore_index=True)
            print str(float(index * runs + run) / float(len(radii) * runs) * 100.0) + '%'

    # Save
    errors.to_csv(filepath, encoding='utf-8')
    # Plot
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.set_xlabel('r')
    ax1.set_ylabel('% found')
    ax1.scatter(errors['r'], errors['num_diff'])

    ax2.set_xlabel('r')
    ax2.set_ylabel('$(r_r - r_f) / r_r$')
    ax2.errorbar(errors['r'], errors['r_diff'], yerr=errors['r_diff_std'], fmt='o')

    ax3.set_xlabel('r')
    ax3.set_ylabel('$x_r - x_f$ (pixels)')
    ax3.errorbar(errors['r'], errors['x_diff'], yerr=errors['x_diff_std'], fmt='o')

    ax4.set_xlabel('r')
    ax4.set_ylabel('$y_r - y_f$ (pixels)')
    ax4.errorbar(errors['r'], errors['y_diff'], yerr=errors['y_diff_std'], fmt='o')

    f.suptitle('Errors in fitting, ' + str(runs) + ' runs')
    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(path)


if __name__ == "__main__":
    main(sys.argv[1:])
