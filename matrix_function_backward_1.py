import numpy as np
from numpy import ndarray

from array_types import ArrayFunction
from derivative import deriv


def matrix_function_backward_1(X: ndarray, W: ndarray, sigma: ArrayFunction) -> ndarray:
    '''
    Computes the derivative of our matrix function w.r.t. the first element.
    '''

    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # backward calculation
    dSdN = deriv(sigma, N)

    # dNdX
    dNdX = np.transpose(W, (1, 0))

    # multiply them together; since dNdX is 1x1 here, order doesn't matter
    return np.dot(dSdN, dNdX)