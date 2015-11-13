import numpy
from pandas import DataFrame
import pandas as pd
from os import path, makedirs, sep
import os
import re


def prepare_dataframe(data_frame, microns_per_pixel):
    # drop unneeded columns
    allowed_cols = ['r', 'dev', 'x', 'y']
    cols = [c for c in data_frame.columns if c in allowed_cols]

    data_frame = data_frame[cols]

    # convert to microns
    data_frame *= microns_per_pixel

    return data_frame


def setup_dir_is_not_exists(filename):
    """

    :param filename:
    :return:
    """
    directory = path.abspath(path.normpath(path.dirname(filename) + sep + 'report'))

    if not path.exists(directory):
        makedirs(directory)

    return directory


def save_circles_to_csv_grouped(data_frame, filename, microns_per_pixel):
    """

    :param data_frame:
    :param filename:
    :param microns_per_pixel:
    """
    if data_frame.empty:
        return

    directory = setup_dir_is_not_exists(filename)

    filename = path.basename(filename)

    data_frame = prepare_dataframe(data_frame, microns_per_pixel)

    # Paths
    file_path_grouped = path.abspath(path.normpath(directory + sep + re.sub("_\d+$", "", filename)))
    report_file_grouped = file_path_grouped + '_grouped_report.csv'
    summary_file_grouped = file_path_grouped + '_grouped_summary.csv'

    # Merge existing
    if path.isfile(report_file_grouped):
        existing_df = DataFrame.from_csv(report_file_grouped)
        data_frame = pd.concat([data_frame, existing_df], ignore_index=True)

    # Delete summary
    if path.isfile(summary_file_grouped):
        os.remove(summary_file_grouped)

    # save DataFrame
    data_frame.to_csv(report_file_grouped, encoding='utf-8')

    # create summary
    summary = generate_summary(data_frame)

    # save
    summary.to_csv(summary_file_grouped, encoding='utf-8')


def save_circles_to_csv(data_frame, filename, microns_per_pixel):
    """
    Save fitted circles to csv files in subdir
    :param data_frame:
    :param filename:
    :param microns_per_pixel:
    """
    if data_frame.empty:
        return

    directory = setup_dir_is_not_exists(filename)

    filename = path.basename(filename)

    data_frame = prepare_dataframe(data_frame, microns_per_pixel)

    # File paths
    file_path = path.abspath(path.normpath(directory + sep + filename))
    report_file = file_path + '_frame.csv'
    summary_file = file_path + '_summary.csv'

    # save DataFrame
    data_frame.to_csv(report_file, encoding='utf-8')

    # create summary
    summary = generate_summary(data_frame)

    # save
    summary.to_csv(summary_file, encoding='utf-8')


def generate_summary(data_frame):
    """

    :param data_frame:
    :return:
    """
    number = len(data_frame.index)
    mean_r = numpy.mean(data_frame['r'])
    error_r = numpy.sqrt(numpy.dot(data_frame['dev'], data_frame['dev'])) / float(number)
    std_r = numpy.std(data_frame['r'])

    data = {
        'R, mean (um)': [mean_r],
        'R, error (um)': [error_r],
        'R, std (um)': [std_r],
        'D, mean (um)': [2.0 * mean_r],
        'D, error (um)': [2.0 * error_r],
        'D, std (um)': [2.0 * std_r],
        'D, std (fraction)': [std_r / mean_r],
        '# particles': [number]
    }
    summary = DataFrame(data)

    return summary
