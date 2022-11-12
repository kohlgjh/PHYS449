# PHYS449

The best results I was able to achieve was ~90% accuracy on training and ~20% on testing with a learning rate of 0.005 and 2000 epochs. However, I found the loss to be quite unstable jumping up and down so by lowering the learning rate to 0.0015 it takes longer to train but is more stable.

I have defined accuracy such that a perfectly matching prediction counts as correct, and a prediction that does not match (even if it is only a single bit) counts as incorrect. Summing the correct predictions and dividing the sum by the total number of predictions (x100%) is the reported accuracy.

The model works pretty well on the training data, but only gets ~1/5 predictions right on the test data. It also gets about 1/5 correct on the reversed test data (b * a instead of a * b). (see the final print statement for this info).

The model takes inputs of one-hot vectors and outputs a one-hot prediction, using softmax to normalize each one-hot vector. 

See loss.png in the plots folder for the plot of training vs test loss. And see training_log.txt for the output at each of the "display_epochs" of the most recent run of the model.

## Dependencies

- numpy
- pytorch
- matplotlib

## Running `main.py`

To run `main.py`, use

```sh
python main.py --param param\param.json --train_size 10000 --test_size 2000 --seed 12345 --device cuda
```
