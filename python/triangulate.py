import numpy as np

def triangulate(P1, pts1, P2, pts2):
    """
    Estimate the 3D positions of points from 2d correspondence
    Args:
        P1:     projection matrix with shape 3 x 4 for image 1
        pts1:   coordinates of points with shape N x 2 on image 1
        P2:     projection matrix with shape 3 x 4 for image 2
        pts2:   coordinates of points with shape N x 2 on image 2

    Returns:
        Pts3d:  coordinates of 3D points with shape N x 3
    """
    num_points = pts1.shape[0]
    pts3d = np.zeros((num_points, 3))

    for i in range(num_points):
            x1, y1 = pts1[i]
            x2, y2 = pts2[i]

            # Construct matrix A
            A = np.array([
                y1 * P1[2, :] - P1[1, :],
                P1[0, :] - x1 * P1[2, :],
                y2 * P2[2, :] - P2[1, :],
                P2[0, :] - x2 * P2[2, :]
            ])

            # Perform SVD
            _, _, Vt = np.linalg.svd(A)
            V = Vt[-1]

            # Normalize to convert to homogeneous coordinates
            pts3d[i, :] = V[:3] / V[3]

    return pts3d



   
