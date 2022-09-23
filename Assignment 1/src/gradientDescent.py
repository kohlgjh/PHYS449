'''Class to implement gradient descent method of solving'''
import numpy as np

class GradientDescentModel():

    def __init__(self, data_array: np.ndarray, learning_rate:float, num_iter:int):
        '''
        Class containing gradient descent model of solution.

        Params
        ----
        data_array: numpy array
            2D array of data of the form [[x1,...xn, y],...]
        learning_rate: float
            learning rate of the gradient descent model
        num_iter: int
            number of iterations to run for gradient descent
        '''
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        
        self.x_vectors = data_array[:,:-1] # each row is different set of x
        self.T_vector = data_array[:,-1] # horizontal vector

        self.PHI_matrix = self._construct_PHI().T # each column is phi vector for a given x vector

        self.w_star = self._perform_gradient_descent()

    def _generate_initial_w(self)->np.ndarray:
        '''Generates an initial guess for w vector of everything being 1'''
        return np.zeros((self.PHI_matrix.shape[0], 1)) + 1

    def _construct_PHI(self)->np.ndarray:
        '''Constructs the capital phi matrix'''
        # our phi_0 is just and our feature map is jsut the identity so 
        #   this becomes just 1 followed by each element of x vector
        phi_0 = np.array([[1]*self.x_vectors.shape[0]]).T
        return np.concatenate([phi_0, self.x_vectors], axis=1)

    def _perform_gradient_descent(self)->np.ndarray:
        '''Performs gradient descent using attributes of learning_rate, and num_iter'''

        current_w = self._generate_initial_w()
        updated_w = current_w.copy()

        for iter_j in range(self.num_iter): # handles number of iterations

            for i in range(self.PHI_matrix.shape[0]): # handles looping through each w element
                sum = 0
                for n in range(self.PHI_matrix.shape[1]): # handles summing
                    sum += (self.T_vector[n] - current_w.T@self.PHI_matrix[:,n:n+1]) * self.PHI_matrix[i,n]

                updated_w[i] = current_w[i] + self.learning_rate * sum

            current_w = updated_w.copy() # update w vector after 1 full iteration

        return current_w
