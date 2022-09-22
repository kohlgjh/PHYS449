import numpy as np

class GradientDescentModel():

    def __init__(self, data_array: np.ndarray, learning_rate:float, num_iter:int):
        '''
        Class containing gradient descent model of solution.

        Params
        ----

        '''
        self.w_star = np.array([1.,1.,1.])