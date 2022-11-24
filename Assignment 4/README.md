# PHYS449

The KL divergence uses an estimate of the probabilities p(x) by counting the number of isntances of a particular configuration and dividing by the total number of samples e.g. 1000 in in.txt.
Taking that as the probability we are able to estiamte the KL divergence. It has high variance due to the random initialization of the model lattices, however, as visible by the plot, on average it reduces as we go towards the true weights.

## Dependencies

- numpy
- matplotlib

## Running `main.py`

To run `main.py`, use

```sh
python main.py data\in.txt --verbose True
```
