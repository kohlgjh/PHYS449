import numpy as np

def avg_coupler(pm_data):
    '''
    Calculates the average of adjacent spins.
    Returns array of the average relations between spins.
    '''
    data_avg_dict = {}
    data_avg_arr = np.empty(pm_data.shape[1])

    for j in range(pm_data.shape[1]):
        if j != pm_data.shape[1] - 1:
            data_avg_arr[j] = np.average(pm_data[:, j] * pm_data[:, j+1])
        else:
            data_avg_arr[j] = np.average(pm_data[:, j] * pm_data[:, 0])

    return data_avg_arr