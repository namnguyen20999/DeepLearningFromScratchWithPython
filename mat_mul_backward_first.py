import numpy as np
from numpy import ndarray


def mat_mul_backward_first(X: ndarray, W: ndarray) -> ndarray:
    '''
    Computes the backward pass of a matrix multiplication with respect to the first argument.
    '''

    # backward pass
    dNdX = np.transpose(W, (1, 0))

    return dNdX