'''Class structure for model'''
import numpy as np
from src.coupler import avg_coupler

class Ising1D():
    '''
    1D Ising model class.
    '''
    def __init__(self, N, num_samples, seed=3141):

        # set random seed for generating weights
        np.random.seed(seed)
        self.weights = np.random.uniform(low=-1., high=1., size=N)

        self.N = N
        self.num_samples = num_samples

        # setup the lattices
        self.generate_lattices()
        

    def generate_lattices(self) -> np.ndarray:
        '''Generates num_samples amount of 1D lattices of shape N'''
        np.random.seed()
        self.lattices = (np.random.randint(0, 2, size=(self.num_samples, self.N)) * 2) - 1

    def equilibrium(self, flips_per_site=100):
        '''
        Lets each lattice go to equilibrium

        Params
        ---
        flips_per_site - average number of flips per site
        '''
        tot_flips = self.N * flips_per_site # total number of flips

        # random indices to attempt to flip
        np.random.seed()
        rand_j = np.random.randint(low=0, high=self.N, size=(self.num_samples, tot_flips))

        # outer loop goes through each lattice, letting each go to equilibrium
        for i in range(self.num_samples):

            # loop through the random indices trying to flip
            for j in rand_j[i, :]:
                # calculate current and new energy at indice j
                current, new = self.get_energy_difference(i, j)

                # comparing energies
                if current >= new:
                    # keep spin if improved or stayed same
                    self.lattices[i, j] = self.lattices[i, j] * -1

                if current < new:
                    # using metropolis algorithm we sometimes take this option
                    if np.random.random(1)[0] < np.exp(-(new - current)):
                        self.lattices[i, j] = self.lattices[i, j] * -1

    def get_energy_difference(self, i, j):
        '''Calculates and returns current and new energy at location j in lattice i'''

        # sum contributions from adjacent spins
        current = self.lattices[i, j] * self.lattices[i, j-1] * self.weights[j-1] * -1
        current += self.lattices[i, j] * self.lattices[i, (1+j-self.N)] * self.weights[1+j-self.N] * -1
        new = current * -1 # flipping of j is simply multiplying by negative 1

        return current, new

    def train_weights(self, train_data, num_epochs, learning_rate, flips_per_site=100,
                      verbose = False):
        '''
        Trains weights based on data set
        '''
        train_coupler_avg = avg_coupler(train_data)

        for epoch in range(num_epochs+1):
            # generate lattices, go to equilibrium, and average couplers
            self.generate_lattices()
            self.equilibrium(flips_per_site=flips_per_site)
            model_coupler_avg = avg_coupler(self.lattices)

            # update weights based on coupler averages
            self.weights = self.weights + learning_rate * (train_coupler_avg - model_coupler_avg)

            # we know weights can't be more than 1 or less than -1 so put a hard stop at -1/+1
            self.weights[np.where(self.weights > 1)] = 1.
            self.weights[np.where(self.weights < -1)] = -1. 

            if verbose:
                if epoch % 25 == 0:
                    current_weights = {}

                    for j in range(self.lattices.shape[1]):
                        if j != self.lattices.shape[1] - 1:
                            current_weights[(j, j+1)] = round(self.weights[j], 2)
                        else:
                            current_weights[(j, 0)] = round(self.weights[j], 2)
                    print(f"Epoch: {epoch}, weights are: {current_weights}\n")


        final_weights = {}

        for j in range(self.lattices.shape[1]):
            if j != self.lattices.shape[1] - 1:
                final_weights[(j, j+1)] = round(self.weights[j], 2)
            else:
                final_weights[(j, 0)] = round(self.weights[j], 2)

        print('Final weights: ', final_weights)

    # def _KL(self, train_data):
    #     '''
    #     tracks KL divergence by estimating probability using number of occurences
    #     divided by total number of lattices
    #     '''
    #     # find number of unique instances in data and current lattices
    #     data_instances = []
    #     model_instances = []

    #     # collect all unique lattices in data
    #     for i in range(train_data.shape[0]):
    #         if train_data[i].tolist() not in data_instances:
    #             data_instances.append(train_data[i].tolist())

    #     #collect all unique lattices in model
    #     for i in range(self.lattices.shape[0]):
    #         if ising.lattices[i].tolist() not in model_instances:
    #             model_instances.append(ising.lattices[i].tolist())

    #     # now calcualte KL
    #     KL = 0
    #     for instance in data_instances:
    #         if instance in model_instances:
    #             p_data = self._p(pm_data, instance)
    #             p_model = self._p(ising.lattices, instance)
    #             KL +=  p_data * np.log(p_data/p_model)
    #     return KL

    def _p(self, data, instance):
        '''Estimates probability of configuration based on number of exact instances in dataset'''
        num_equal = 0
        for lattice in data:
            if np.array_equal(lattice, instance): num_equal += 1

        return num_equal/data.shape[0]