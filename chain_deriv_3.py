import numpy as np
from numpy import ndarray
from nested_function import Chain
from derivative import deriv


def sigmoid(x: ndarray) -> ndarray:
    """
    Apply the sigmoid function to each element in the input array.
    """
    return 1 / (1 + np.exp(-x))


def chain_deriv_3(chain: Chain, input_range: ndarray) -> ndarray:
    """
    Use the chain rule to compute the derivative of three nested functions:
    f3((f2(f1(x)))' = f3'(f2(f1(x))) * f2'(f1(x)) * f1'(x)
    """

    assert len(chain) == 3, "This function requires 'Chain' objects of length 3"

    assert input_range.ndim == 1, (
        "Function requires a 1 dimensional ndarray as input_range"
    )

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    # f1(x)
    f1_of_x = f1(input_range)

    # f2(f1(x))
    f2_of_x = f2(f1_of_x)

    # df3/du
    df3du = deriv(f3, f2_of_x)

    # df2du
    df2du = deriv(f2, f1_of_x)

    # df1dx
    df1dx = deriv(f1, input_range)

    # Multiplying these quantities together at each point
    return df1dx * df2du * df3du
