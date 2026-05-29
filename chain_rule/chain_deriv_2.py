import numpy as np
from numpy import ndarray
from chain_rule.nested_function import Chain
from core.derivative import deriv


def sigmoid(x: ndarray) -> ndarray:
    """
    Apply the sigmoid function to each element in the input array.
    """
    return 1 / (1 + np.exp(-x))


def chain_deriv_2(chain: Chain, input_range: ndarray) -> ndarray:
    """
    Use the chain rule to compute the derivative of two nested functions:
    (f2(f1(x))' = f2'(f1(x)) * f1'(x)
    """

    assert len(chain) == 2, "This function requires 'Chain' objects of length 2"

    assert input_range.ndim == 1, (
        "Function requires a 1 dimensional ndarray as input_range"
    )

    f1 = chain[0]
    f2 = chain[1]

    # df1/dx
    f1_of_x = f1(input_range)

    # df1/du
    df1dx = deriv(f1, input_range)

    # df2/du(f1(x))
    df2du = deriv(f2, f1_of_x)

    # Multiplying these quantities together at each point
    return df1dx * df2du
