'''Class to generate the training/test data, and interlace binary numbers for the input'''

import numpy as np
import re

def interlace(in1:np.ndarray, in2:np.ndarray) -> np.ndarray:
    '''
    Interlaces in1 and in2
    
    Example: 
    in1 = [[[0 1], [1 0], ...], ...]
    in2 = [[[1 0], [1 0], ...], ...]
    returns: [[[0 1], [1 0], [1 0], [1 0]], ...]
    '''
    laced = np.empty((in1.shape[0], in1.shape[1]*2, 2), dtype=int)

    for i in range(in1.shape[0]):
        for j in range(in1.shape[1]):
            laced[i, 2*j] = in1[i,j]
            laced[i, (2*j)+1] = in2[i,j]

    return laced

def generate_train_test(train_size, test_size, seed) -> np.ndarray:
    '''
    Generates the training and testing data given a random seed.

    We use one-hot defintiions of:
    0 = [0 1], 1 = [1 0]

    Returns:
    A_train - 2D array where each row is separated digits of 8-bit binary number
    B_train - 2D array where each row is separated digits of 8-bit binary number
    C_train - 2D array where each row is separated digits of a 16-bit binary number of A*B
    A_test - ...
    B_test - ...
    C_test - ...
    '''

    total_size = train_size + test_size

    # generate the random integers 0-255
    np.random.seed(seed)
    ints = np.random.randint(0, 256, size=total_size*2)
    A = ints.copy()[0:total_size].astype(np.int64)
    B = ints.copy()[total_size:].astype(np.int64)
    C = A*B.astype(np.int64)

    # arrays to hold the binary-representation-strings of the converted integers
    Astr, Bstr, Cstr = np.empty(len(A), dtype=object), np.empty(len(A), dtype=object), np.empty(len(A), dtype=object)

    for i in range(len(A)):
        Astr[i] = np.binary_repr(A[i], width=8)
        Bstr[i] = np.binary_repr(B[i], width=8)
        Cstr[i] = np.binary_repr(C[i], width=16)

    # arrays to hold the 0's and 1's that come from splitting the strings of binary-representation
    Aarr, Barr, Carr = np.empty((len(A),8,2), dtype=object), np.empty((len(A),8,2), dtype=object), np.empty((len(A),16,2), dtype=object)

    for i in range(len(A)):
        for j, binary in enumerate(np.flip(np.array(re.split('', Astr[i])[1:-1], dtype=int))):
            Aarr[i, j] = [1, 0] if binary == 1 else [0, 1]

        for j, binary in enumerate(np.flip(np.array(re.split('', Bstr[i])[1:-1], dtype=int))):
            Barr[i, j] = [1, 0] if binary == 1 else [0, 1]

        for j, binary in enumerate(np.flip(np.array(re.split('', Cstr[i])[1:-1], dtype=int))):
            Carr[i, j] = [1, 0] if binary == 1 else [0, 1]
    
    return Aarr[0:train_size], Barr[0:train_size], Carr[0:train_size], Aarr[train_size:], Barr[train_size:], Carr[train_size:]

# TESTING

# a,b,c,d,e,f = generate_train_test(100, 10, 12345)
# print(a[0])
# print(b[0])
# print(interlace(a,b)[0])