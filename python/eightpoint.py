import numpy as np
from numpy.linalg import svd
from refineF import refineF

def eightpoint(pts1, pts2, M):
    """
    eightpoint:
        pts1 - Nx2 matrix of (x,y) coordinates
        pts2 - Nx2 matrix of (x,y) coordinates
        M    - max(imwidth, imheight)
    """
    
    # Implement the eightpoint algorithm
    # Generate a matrix F from correspondence '../data/some_corresp.npy'
    # Normalize points
    # Normalize points
    # Define the transformation matrix T
    T = np.array([
        [1 / M, 0, 0],
        [0, 1 / M, 0],
        [0, 0, 1]
    ])


    pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))       
    # Assuming pts1 and pts2 are numpy arrays with shape (N, 3)
    # Apply the transformation
    pts1 = pts1 @ T.T
    pts2 = pts2 @ T.T

    len = pts1.shape[0]
    A = np.zeros((len, 9))

    for i in range(3):  # Note that Python uses 0-based indexing
        for j in range(3):
            col = (i * 3) + j
            A[:, col] = pts1[:, i] * pts2[:, j]
    
    

    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Set the smallest singular value to 0 to enforce rank 2
    U, D, Vt = np.linalg.svd(F)
    D = np.diag(D)
    D[-1, -1] = 0
    F = U @ D @ Vt

    # Refine the solution using local minimization (refineF function)
    F = refineF(F, pts1, pts2)

    # Apply the transformation matrix T
    F = T.T @ F @ T

    return F
