import numpy as np
import cv2
def get_disparity(im1, im2, maxDisp, windowSize):
    """
    creates a disparity map from a pair of rectified images im1 and
    im2, given the maximum disparity MAXDISP and the window size WINDOWSIZE.
    """
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)

    # Initialize disparity map and minimum disparity map
    dispM = np.zeros_like(im1)
    minDispM = np.full_like(im1, np.inf)

    # Define the mask (kernel for convolution)
    mask = np.ones((windowSize, windowSize))

    # Iterate over disparity values
    for d in range(maxDisp + 1):
        # Translate im2 by d pixels to the right
        translatedIm2 = np.roll(im2, d, axis=1)
        translatedIm2[:, :d] = 255  # Fill values to the left

        # Compute squared differences and convolve with the mask
        currentDispM = cv2.filter2D((im1 - translatedIm2) ** 2, -1, mask, borderType=cv2.BORDER_CONSTANT)

        # Update disparity map
        mask_update = currentDispM < minDispM
        dispM[mask_update] = d
        minDispM = np.minimum(minDispM, currentDispM)

    return dispM

