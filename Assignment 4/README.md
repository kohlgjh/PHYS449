# PHYS449

The KL divergence uses an estimate of the probabilities p(x) by counting the number of isntances of a particular configuration and dividing by the total number of samples e.g. 1000 in in.txt.
Taking that as the probability we are able to estiamte the KL divergence. It has high variance due to the random initialization of the model lattices, however, as visible by the plot, on average it reduces as we go towards the true weights.


The 1D Ising model is implemented using a Monte Carlo algorithm where randomly, sites within the lattice are attempted to be flipped. Them the resutlant energy is compared with the energy it had before it was it was flipped. If the the new energy is lower (more stable) then we keep the flip. If the new energy is greater (less stable) then we employ a Metropolis approach and only keep this worse less stable flip with proabbility given by exp{-(new - current)}. Thus allowing the algorithm to explore possible configurations, but tend to more stable ones.

## Dependencies

- numpy
- matplotlib

## Running `main.py`

To run `main.py`, use

```sh
python main.py data\in.txt --verbose True
```
