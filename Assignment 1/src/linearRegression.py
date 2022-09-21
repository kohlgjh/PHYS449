'''Class for handling the linear regression solution'''
import numpy as np

class LinearRegressionModel():

    def __init__(self, data_array: np.ndarray):
        '''
        Solver for linear regression model solution

        Params
        ---
        data_array: 2D numpy array of data in form: [[x1, x2,..xN, y],...]
        '''
        self.x_vectors = data_array[:,:-1] # each row is different set of x
        self.T_vector = data_array[:,-1:] # vertical vector

        self.PHI_matrix = self._construct_PHI()
        self.w_star = self._construct_w_star()

    def _construct_PHI(self)->np.ndarray:
        '''Constructs the capital phi matrix'''
        # our phi_0 is just and our feature map is jsut the identity so 
        #   this becomes just 1 followed by each element of x vector
        phi_0 = np.array([[1]*self.x_vectors.shape[0]]).T
        return np.concatenate([phi_0, self.x_vectors], axis=1)

    def _construct_w_star(self):
        '''Constructs the optimal w matrix'''
        return np.linalg.inv(self.PHI_matrix.T@self.PHI_matrix)@self.PHI_matrix.T@self.T_vector