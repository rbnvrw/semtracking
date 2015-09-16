import numpy as np
from pandas import DataFrame


def save_circles_to_csv(dataframe, filename):
    # save dataframe
    dataframe.to_csv(filename + '_frame.csv', encoding='utf-8')

    # create summary
    mean_r = np.mean(dataframe['r'])
    dev_r = np.std(dataframe['r'])

    data = {'mean_r': [mean_r], 'dev_r': [dev_r]}
    summary = DataFrame(data)

    # save
    summary.to_csv(filename + '_summary.csv', encoding='utf-8')
