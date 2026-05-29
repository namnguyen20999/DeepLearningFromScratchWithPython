import numpy as np
from numpy import ndarray

from core.array_types import ArrayFunction
from core.derivative import deriv
from matrix_ops.matrix_function_forward_sum import matrix_function_forward_sum


def matrix_function_backward_sum_1(X: ndarray, W: ndarray, sigma: ArrayFunction) -> ndarray:
    '''
    Compute derivative of matrix function with a sum with respect to the
    first matrix input.
    '''
    assert X.shape[1] == W.shape[0]

    # Forward pass — recompute intermediate values needed for the backward pass
    N = np.dot(X, W)   # (batch, out_feat): pre-activation
    S = sigma(N)       # (batch, out_feat): post-activation
    L = np.sum(S)      # scalar loss

    # dL/dS: the loss is just sum(S), so every element of S contributes equally — all 1s
    dLdS = np.ones_like(S)  # same shape as S

    # dS/dN: how much each pre-activation value N affects S element-wise.
    # This is the derivative of sigma at every point in N (numerical central difference).
    dSdN = deriv(sigma, N)  # same shape as N

    # dL/dN: chain rule — dL/dS * dS/dN element-wise.
    # Since dLdS is all 1s this equals dSdN, but we keep the step explicit for clarity.
    dLdN = dLdS * dSdN  # same shape as N

    # dN/dX: N = X @ W, so the gradient of N w.r.t. X is W^T.
    # (This comes from matrix calculus: d(XW)/dX = W^T)
    dNdX = np.transpose(W, (1, 0))  # (out_feat, in_feat)

    # dL/dX: chain rule across the matmul — dL/dN @ dN/dX.
    # dLdN is (batch, out_feat), dNdX is (out_feat, in_feat) → result is (batch, in_feat),
    # matching the shape of X so each element of X gets its own gradient.
    # Note: dLdN equals dSdN here, so we use dSdN directly.
    dLdX = np.dot(dSdN, dNdX)

    return dLdX


def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    np.random.seed(190204)

    X = np.random.randn(3, 3)
    W = np.random.randn(3, 2)

    print("X:")
    print(X)
    print("L:")
    print(round(matrix_function_forward_sum(X, W, sigmoid), 4))
    print()
    print("dLdX:")
    print(matrix_function_backward_sum_1(X, W, sigmoid))
