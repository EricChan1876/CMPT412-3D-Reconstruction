import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
from PIL import Image
from eightpoint import eightpoint
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate
from displayEpipolarF import displayEpipolarF
from epipolarMatchGUI import epipolarMatchGUI
import os 


#1. Load the two images and the point correspondences from someCorresp.npy
# Load images and points
img1 = cv2.imread('../data/im1.png')
img2 = cv2.imread('../data/im2.png')
pts = np.load('../data/someCorresp.npy', allow_pickle=True).tolist()
pts1 = pts['pts1']      #(x,y) coordinate in piture 1
pts2 = pts['pts2']      #(x,y) coordinate in piture 2
M = pts['M']            #scalar parameter

# write your code here
R1, t1 = np.eye(3), np.zeros((3, 1))
R2, t2 = np.eye(3), np.zeros((3, 1))

#2. Run eightpoint to compute the fundamental matrix F
F = eightpoint(pts1, pts2, M)
#displayEpipolarF(img1, img2, F)
epipolarMatchGUI(img1,img2,F)

#3. Load the points in image 1 contained in templeCoords.npy and run your epipolarCorrespondences on them to get the corresponding points in image 
pts = np.load('../data/templeCoords.npy', allow_pickle=True).tolist()
pts1=pts['pts1']
pts2 = np.zeros_like(pts1)

# Iterate through each point in pts1 and find its corresponding point in im2
for i in range(pts1.shape[0]):
    pts2[i] = epipolarCorrespondence(img1, img2, F, np.array([pts1[i]]))


#4. Load intrinsics.npy and compute the essential matrix E.
K = np.load('../data/intrinsics.npy', allow_pickle=True).tolist()
K1=K['K1']
K2 = K ['K2']
E = essentialMatrix(F, K1, K2)


#5. Compute the first camera projection matrix P1 and use camera2.py to compute the four candidates for P2

P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2Candidates = camera2(E)

#6 Run your triangulate function using the four sets of camera matrix candidates, the points from templeCoords.npy and their computed correspondences.
min_distance = 1e12
min_distance1 = 1e12
min_distance2 = 1e12

for i in range(4):
    P2_candidate = P2Candidates[:, :, i]

    # Adjust P2 if the determinant is not 1
    if np.linalg.det(P2_candidate[:3, :3]) != 1:
        P2_candidate = K2 @ P2_candidate

    # Triangulate points
    pts3d_candidate = triangulate(P1, pts1, P2_candidate, pts2)

    # Project points back to image planes
    x1 = P1 @ np.hstack((pts3d_candidate, np.ones((pts3d_candidate.shape[0], 1)))).T
    x2 = P2_candidate @ np.hstack((pts3d_candidate, np.ones((pts3d_candidate.shape[0], 1)))).T

    # Normalize points
    epsilon = 1e-6  # A small threshold to prevent division by very small numbers
    x1[:, x1[2, :] > epsilon] /= x1[2, x1[2, :] > epsilon]
    x2[:, x2[2, :] > epsilon] /= x2[2, x2[2, :] > epsilon]

    # Check if all points are in front of the camera
    if np.all(pts3d_candidate[:, 2] > 0):
        distance1 = np.linalg.norm(pts1 - x1[:2, :].T) / pts3d_candidate.shape[0]
        distance2 = np.linalg.norm(pts2 - x2[:2, :].T) / pts3d_candidate.shape[0]
        distance = distance1 + distance2

        # Update if this is the minimum distance so far
        if distance < min_distance:
            min_distance = distance
            min_distance1 = distance1
            min_distance2 = distance2
            pts3d = pts3d_candidate
            P2 = P2_candidate


print(f'Min pts1 error: {min_distance1}')
print(f'Min pts2 error: {min_distance2}')

# Plot the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts3d_candidate[:, 0], pts3d_candidate[:, 1], pts3d_candidate[:, 2], c='k', marker='.')

# Set equal scaling
ax.set_box_aspect([1,1,1])  # matplotlib 3.1.0 and later


plt.show()

#9. Save your computed rotation matrix (R1, R2) and translation (t1, t2) to the file ../result/extrinsics.npy. These extrinsic parameters will be used in the next section.
# Compute the rotation matrices and translation vectors
R1, _, _, _ = np.linalg.lstsq(K1, P1[:3, :3], rcond=None)
t1, _, _, _ = np.linalg.lstsq(K1, P1[:, 3], rcond=None)
R2, _, _, _ = np.linalg.lstsq(K2, P2[:3, :3], rcond=None)
t2, _, _, _ = np.linalg.lstsq(K2, P2[:, 3], rcond=None)

# Make sure that R1, R2 are 3x3 matrices and t1, t2 are 3x1 vectors
R1 = R1.T
t1 = t1.ravel()
R2 = R2.T
t2 = t2.ravel()



os.makedirs('../results/extrinsics', exist_ok=True)
# save extrinsic parameters for dense reconstruction
np.save('../results/extrinsics', {'R1': R1, 't1': t1, 'R2': R2, 't2': t2})

print("save completed")