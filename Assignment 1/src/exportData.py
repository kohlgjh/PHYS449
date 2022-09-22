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
    with open(path, 'w') as save_file:
        for w_analytic_i in w_analytic:
            save_file.write(f'{w_analytic_i[0]:.4f}\n')
        save_file.write('\n')
        for w_gd_i in w_gd:
            save_file.write(f'{w_gd_i:.4f}\n')