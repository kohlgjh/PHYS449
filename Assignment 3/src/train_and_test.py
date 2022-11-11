'''Functions for training and testing the model'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import os

def generate_accuracy(model, loader):
    '''Evaluates the accuracy of the model on the passed data. Returns % accuracy'''
    labels = loader.dataset[:][1].cpu().detach().numpy().astype(int)
    data = loader.dataset[:][0]
    preds = np.round(model.forward(data, data.shape[0]).cpu().detach().numpy()).astype(int)
    
    # compare each element of pred and label and add to sum trackign correct predictions
    with torch.no_grad():
        pred_sum = 0
        for label, pred in zip(labels, preds):
            if np.array_equal(label, pred): pred_sum += 1
    return pred_sum/labels.shape[0] * 100


def train_and_test(model, train_loader, test_loader, params) -> np.ndarray:
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

    model.reset() # reset parameters

    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))
    loss = nn.CrossEntropyLoss()

    for epoch in range(int(params['num_epochs'])):

        for (train, train_targets), (test, test_targets) in zip(train_loader, test_loader):

            obj_val = loss(model.forward(train, train_loader.batch_size), train_targets)
            
            optimizer.zero_grad()
            obj_val.backward()
            optimizer.step()
            obj_vals.append(obj_val.item())
                    
            # as it trains check how well it tests
            with torch.no_grad(): 
                # don't track calculations in the following scope for the purposes of gradients
                cross_val = loss(model.forward(test, test_loader.batch_size), test_targets)
                cross_vals.append(cross_val.item())

        if (epoch+1) % int(params["display_epochs"]) == 0:
            # evaluate accuracy
            test_accuracy = generate_accuracy(model, test_loader)
            train_accuracy = generate_accuracy(model, train_loader)

            # string of current loss and accuracy for train/test data
            updates = ('Epoch [{}/{}]\t Training Loss: {:.3f}  Training Accuracy: {:.1f}% \t Test Loss: {:.3f}  Test Accuracy: {:.1f}%'
                    .format(epoch+1, int(params['num_epochs']), obj_val.item(), train_accuracy, cross_val.item(), test_accuracy))
            
            log += '\n'+ updates # update the log with the training process
            
            if params["verbose"] != False: # only print if verbosity is True
                print(updates)
    
    print('Final training loss: {:.4f}'.format(obj_vals[-1]))
    print('Final test loss: {:.4f}'.format(cross_vals[-1]))

    # save the logging of progress
    with open(os.path.join(os.getcwd(), 'training_log.txt'), 'w') as log_file:
        log_file.write(log)

    accuracy = generate_accuracy(model, test_loader)

    return obj_vals, cross_vals, accuracy
