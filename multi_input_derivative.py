from numpy import ndarray

from array_types import ArrayFunction
from derivative import deriv


def multiple_inputs_add_backward(x: ndarray, y: ndarray, sigma: ArrayFunction) -> float:
    """
    Computes the derivative of this simple function with respect to both inputs
    """

    # Compute 'forward pass'
    a = x + y

    # Computer the derivative
    dsda = deriv(sigma, a)

    dadx, dady = 1, 1

    return dsda * dadx, dsda * dady
