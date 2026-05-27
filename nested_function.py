from numpy import ndarray
from array_types import Chain


def chain_length_2(chain: Chain, a: ndarray) -> ndarray:
    """
    Evaluates two functions in a row, in a "Chain".
    """
    assert len(chain) == 2, "Length of input 'chain' should be 2."

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(a))
