import numpy as np
import cv2

def epipolarCorrespondence(im1, im2, F, pts1):
    """
    Args:
        im1:    Image 1
        im2:    Image 2
        F:      Fundamental Matrix from im1 to im2
        pts1:   coordinates of points in image 1
    Returns:
        pts2:   coordinates of points in image 2
    """
    
    pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts1 = pts1.T
    
    len = pts1.shape[0]
    l = F @ pts1
    l /= -l[1, :]  # Normalizing lines
   
    # Round pts1 to integer for indices
    pts1 = np.round(pts1).astype(int)

    # Crop patch 1
    x, y = pts1[0, 0], pts1[1, 0]

    # Indexing into im1 using x and y
    patches1 = im1[y - 3:y + 4, x - 3:x + 4, :]

    # Define search range for matching patch
    startX = max(0, pts1[0] - 10)
    endX = min(im1.shape[1], pts1[0] + 10)
    
    
    
    pts2 = [pts1[0], int(l[0] * pts1[0] + l[2])]

    # Initialize minDistance with a large value
    minDistance = float('inf')

    startX = int(startX)
    endX = int(endX)
    for x in range(startX, endX):
        y = int(l[0] * x + l[2])
        pt2 = [x, y]

        # Check if the patch is within the bounds of the image
        if y - 3 >= 0 and y + 4 <= im2.shape[0] and x - 3 >= 0 and x + 4 <= im2.shape[1]:
            # Extract patch from im2 and calculate distance
            patches2 = im2[y - 3:y + 4, x - 3:x + 4]
            distance = np.sqrt(np.sum((patches2 - patches1) ** 2))

            # Update minimum distance and pts2
            if distance < minDistance:
                minDistance = distance
                pts2 = pt2
        
    
    pts2 = np.array(pts2).reshape(1, -1)
    return pts2
