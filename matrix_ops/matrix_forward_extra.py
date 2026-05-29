import numpy as np
from numpy import ndarray

from core.array_types import ArrayFunction


def matrix_forward_extra(X: ndarray, W: ndarray, sigma: ArrayFunction)-> ndarray:
    '''
    Computes the forward pass of a function involving matrix multiplication, one extra function.
    '''

    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    return S