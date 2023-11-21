import numpy as np
from scipy.linalg import svd

def estimate_pose(x, X):
    """
    computes the pose matrix (camera matrix) P given 2D and 3D
    points.
    
    Args:
        x: 2D points with shape [2, N]
        X: 3D points with shape [3, N]
    """
    A = np.zeros((2 * x.shape[1], 12))

    # Construct matrix A
    for i in range(x.shape[1]):
        xi, yi = x[:, i]
        Xi, Yi, Zi = X[:, i]

        A[2 * i] = [-Xi, -Yi, -Zi, -1, 0, 0, 0, 0, xi * Xi, xi * Yi, xi * Zi, xi]
        A[2 * i + 1] = [0, 0, 0, 0, -Xi, -Yi, -Zi, -1, yi * Xi, yi * Yi, yi * Zi, yi]

    # Perform SVD
    _, _, Vt = np.linalg.svd(A)

    # Extract the projection matrix P
    P = Vt[-1].reshape((4, 3)).T
    
    return P
