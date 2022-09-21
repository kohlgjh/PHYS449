'''Functions to handle exporting the results to an output text file [.out]'''
import numpy as np
import os

def export(path:os.path, w_analytic:np.ndarray, w_gd:np.ndarray)->None:
    '''Exports an output file of the analytic and gradient descent solutions in the
    the following format:

    w_analytic1
    w_analytic2
    w_analytic3

    w_gd1
    w_gd2
    w_gd3

    Params
    ---
    path: 
        the path to the output file
    w_analytic: numpy array
        array of the analytic solution vector
    w_gd: numpy array
        array of the gradient descent solution vector
    '''