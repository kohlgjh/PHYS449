'''Loads data in and converts to arrays of +1 and -1'''
import numpy as np
import os

def load_data(args) -> np.ndarray:
    '''Loads data and returns array of +1/-1'''
    cwd = os.getcwd()

    #import
    pm_data_str = np.loadtxt(os.path.join(cwd, str(args.data)), dtype=str)

    # separate string into characters
    pm_data_sep = np.empty((pm_data_str.shape[0], len(pm_data_str[0])), dtype=str)
    for i in range(len(pm_data_str)):
        pm_data_sep[i] = list(pm_data_str[i])

    # locations of plus minus
    p_loc = np.where(pm_data_sep == '+')
    m_loc = np.where(pm_data_sep == '-')

    # convert plus/minus to +1/-1
    pm_data = np.empty(pm_data_sep.shape)
    pm_data[p_loc] = 1.
    pm_data[m_loc] = -1.

    return pm_data