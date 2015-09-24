from numpy import std, mean
from pandas import DataFrame
from os import path, makedirs, sep


def save_circles_to_csv(data_frame, filename, microns_per_pixel):
    """
    Save fitted circles to csv files in subdir
    :param data_frame:
    :param filename:
    :param microns_per_pixel:
    """
    directory = path.abspath(path.normpath(path.dirname(filename) + sep + 'report'))

    if not path.exists(directory):
        makedirs(directory)

    filename = path.basename(filename)

    # convert to microns
    data_frame *= microns_per_pixel

    # save DataFrame
    file_path = path.abspath(path.normpath(directory + sep + filename))
    data_frame.to_csv(file_path + '_frame.csv', encoding='utf-8')

    # create summary
    mean_r = mean(data_frame['r'])
    dev_r = std(data_frame['r'])

    data = {
        'Mean radius (um)': [mean_r],
        'Dev. in radius (um)': [dev_r],
        'Mean diameter (um)': [2 * mean_r],
        'Dev. in diameter (um)': [2 * dev_r],
        'Dev. in diameter (fraction)': [dev_r / mean_r]
    }
    summary = DataFrame(data)

    # save
    summary.to_csv(file_path + '_summary.csv', encoding='utf-8')
