import numpy as np

def rectify_pair(K1, K2, R1, R2, t1, t2):
    """
    takes left and right camera paramters (K, R, T) and returns left
    and right rectification matrices (M1, M2) and updated camera parameters. You
    can test your function using the provided script testRectify.py
    """
    # YOUR CODE HERE

    M1, M2, K1n, K2n, R1n, R2n, t1n, t2n = None, None, None, None, None, None, None, None

    #1. Compute the optical center c1 and c2 of each camera by c = −(KR)^{−1}(Kt).
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1.reshape(-1, 1))
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2.reshape(-1, 1))
    
    #2. Compute the new rotation matrix 
    # Calculate unit vector r1
    r1 = np.abs(c1 - c2).ravel() / np.linalg.norm(c1 - c2)

    # Normalize r1
    #r1 = r1.reshape(-1)  # Convert to 1D array if it's not already

    # Calculate new rotation vectors for R1
    r21 = np.cross(R1[2, :], r1)
    r21 /= np.linalg.norm(r21)
    r31 = np.cross(r1, r21)
    r31 /= np.linalg.norm(r31)
    R1n = np.array([r1, r21, r31]).T  # Transpose to make each row a vector

    # Calculate new rotation vectors for R2
    r22 = np.cross(R2[2, :], r1)
    r22 /= np.linalg.norm(r22)
    r32 = np.cross(r1, r22)
    R2n = np.array([r1, r22, r32]).T  # Transpose to make each row a vector
    
    #3. Compute the new intrinsic parameter
    K1n = K2;
    K2n = K2;
    
    #4. Compute the new translation:
    t1n = -R1n @ c1
    t2n = -R2n @ c2

    #5. Compute the rectification matrix
    M1 = (K1n @ R1n) @ np.linalg.inv(K1 @ R1)
    M2 = (K2n @ R2n) @ np.linalg.inv(K2 @ R2)





    return M1, M2, K1n, K2n, R1n, R2n, t1n, t2n

