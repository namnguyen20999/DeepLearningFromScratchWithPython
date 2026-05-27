from numpy import ndarray
import numpy as np


def matmul_forward(X: ndarray, W: ndarray) -> ndarray:
    """
    Computes the forward pass of a matrix multiplication.
    """

    assert X.shape[1] == W.shape[0], f"""
    For matrix multiplication, the number of columns in the first array should match
    the number of rows in the second; instead the number of columns in the first array
    is {X.shape[1]} and the number of rows in the second array is {W.shape[0]}.
    """

    # matrix multiplication
    N = np.dot(X, W)

    return N


if __name__ == "__main__":
    np.set_printoptions(precision=4)

    # 1. Square matrices (2x2 @ 2x2 -> 2x2)
    X1 = np.array([[1, 2], [3, 4]], dtype=float)
    W1 = np.array([[5, 6], [7, 8]], dtype=float)
    print("=== Square (2x2 @ 2x2) ===")
    print(matmul_forward(X1, W1))

    # 2. Batch of samples x features (3x4 @ 4x2 -> 3x2)
    #    Typical in a neural network layer: 3 samples, 4 input features, 2 neurons
    np.random.seed(0)
    X2 = np.random.randn(3, 4)
    W2 = np.random.randn(4, 2)
    print("\n=== Batch (3x4 @ 4x2) ===")
    print("X:\n", X2)
    print("W:\n", W2)
    print("Out:\n", matmul_forward(X2, W2))

    # 3. Single sample (1x3 @ 3x1 -> 1x1, i.e. dot product)
    X3 = np.array([[1, 2, 3]], dtype=float)
    W3 = np.array([[4], [5], [6]], dtype=float)
    print("\n=== Dot product (1x3 @ 3x1) ===")
    print(matmul_forward(X3, W3))

    # 4. Dimension mismatch — should raise AssertionError
    print("\n=== Dimension mismatch (2x3 @ 2x2) ===")
    try:
        matmul_forward(np.ones((2, 3)), np.ones((2, 2)))
    except AssertionError as e:
        print("AssertionError caught:", e)
