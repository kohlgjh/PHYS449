'''Functions for training + testing the model'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def generate_accuracy(model, input, labels):
    # calculate the accuracy of the model prediction on test values

    results = np.exp(model.forward(input).detach().numpy())

    eval = np.empty((results.shape[0], 2)) # setup up arrays to hold model prediction
    for i, result in enumerate(results):
        eval[i, 0] = np.argmax(result)*2
        eval[i, 1] = np.max(result)

    sum = 0 # sum up the number of incorrect predictions
    for result, label in zip(eval[:,0].astype(int), labels.detach().numpy()*2):
        if result != label:
            sum += 1
    return (1-sum/eval.shape[0])*100


def train_and_test(model, train_input, train_labels, test_input, test_labels,
                   num_epochs, display_epochs, verbose, learning_rate) -> np.ndarray:
    '''
    Function that handles the training and testing of the neaural net
    
    Returns
    ---
    obj_vals: numpy array
        the loss value at each epoch of the training data
    cross_vals: numpy array
        the loss value at each epoch of the test data
    accuracy: float
        percentage accuracy of the model when tested
    '''
    obj_vals= []
    cross_vals= []
    log = ''

    model.reset() # reset your parameters

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        obj_val = F.nll_loss(model.forward(train_input), train_labels)
        
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        obj_vals.append(obj_val.item())
                
        # as it trains check how well it tests
        with torch.no_grad(): 
            # don't track calculations in the following scope for the purposes of gradients
            cross_val = F.nll_loss(model.forward(test_input), test_labels)
            cross_vals.append(cross_val)

        if (epoch+1) % display_epochs == 0:
            if verbose:
                test_accuracy = generate_accuracy(model, test_input, test_labels)
                train_accuracy = generate_accuracy(model, train_input, train_labels)
                updates = ('Epoch [{}/{}]\t Training Loss: {:.4f}  Training Accuracy: {:.1f}% \t Test Loss: {:.4f}  Test Accuracy: {:.1f}%'
                      .format(epoch+1, num_epochs, obj_val.item(), train_accuracy, cross_val.item(), test_accuracy))
                print(updates)
                log += '\n'+ updates

    # save the logging of progress
    with open(os.path.join(os.getcwd(), 'training_log.txt'), 'w') as log_file:
        log_file.write(log)

    print('Final training loss: {:.4f}'.format(obj_vals[-1]))
    print('Final test loss: {:.4f}'.format(cross_vals[-1]))

    accuracy = generate_accuracy(model, test_input, test_labels)
    print(f"\nModel prediction accuracy: {accuracy:.1f}%")

    return obj_vals, cross_vals, accuracy
