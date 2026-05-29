from numpy import ndarray
import numpy as np
from array_types import ArrayFunction


def matrix_function_forward_sum(X: ndarray, W: ndarray, sigma: ArrayFunction) -> float:
    '''
    Computing the result of the forward pass of this function with
    input ndarrays X and W and function sigma.
    '''

    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # sum all the elements
    L = np.sum(S)

    return L