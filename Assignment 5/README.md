# PHYS449

I've locked in the hyperparameters for this assignment as the combination I found works extremely well 
and extremely quickly so there is no need to change them.

## Dependencies

- numpy
- matplotlib
- torch

## Running `main.py`

To run `main.py`, use

```sh
python main.py -o results -n 100 -d cuda -v True
```
Arguments are:

--output or -o: Relative path to output directory

--number or -n: number of integer sample images to produce

--device or -d: the device to use: cuda or cpu

--verbose or -v: True to turn on high verbosity
