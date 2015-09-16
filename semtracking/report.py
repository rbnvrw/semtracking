import numpy as np
from pandas import DataFrame
import os


def save_circles_to_csv(dataframe, filename, microns_per_pixel):
    directory = os.path.dirname(filename) + '/report/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.basename(filename)

    # convert to microns
    dataframe *= microns_per_pixel

    # save dataframe
    dataframe.to_csv(directory + filename + '_frame.csv', encoding='utf-8')

    # create summary
    mean_r = np.mean(dataframe['r'])
    dev_r = np.std(dataframe['r'])

    data = {
        'Mean radius (um)': [mean_r],
        'Dev. in radius (um)': [dev_r],
        'Mean diameter (um)': [2*mean_r],
        'Dev. in diameter (um)': [2*dev_r],
        'Dev. in diameter (fraction)': [dev_r/mean_r]
    }
    summary = DataFrame(data)

    # save
    summary.to_csv(directory + filename + '_summary.csv', encoding='utf-8')
