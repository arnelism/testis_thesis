import math
from typing import Literal

import numpy as np
import scipy
from skimage.morphology import binary_dilation

from image_utils import CH_BORDER, CH_TUBULE, CH_BACKGR
import skimage


def get_clean_image(seg: np.ndarray, mode: Literal["argmax", "otsu"], black_bg=True) -> np.ndarray:
    """
    Initial segmentation map from model is not confident. Separate the border, tubule and bg channels
    more cleanly.
    If mode=argmax, each pixel will belong to one (and only one) of the classes, no overlapping between classes
    If mode=otsu, perform otsu thresholding independently on each channel
    If black bg, set background channel to 0 everywhere
    """
    clean = seg.copy()
    if mode == "argmax":
        # choose the highest value channel for each pixel
        mask = np.argmax(seg, axis=2)
        # set the pixel value to 255 in that channel, other channels set to 0
        clean[:, :, CH_BORDER] = mask == CH_BORDER
        clean[:, :, CH_TUBULE] = mask == CH_TUBULE
        clean[:, :, CH_BACKGR] = mask == CH_BACKGR
    else:
        clean[:, :, CH_BORDER] = perform_otsu(clean[:, :, CH_BORDER])
        clean[:, :, CH_TUBULE] = perform_otsu(clean[:, :, CH_TUBULE])
        clean[:, :, CH_BACKGR] = perform_otsu(clean[:, :, CH_BACKGR])

    if black_bg:
        clean[:, :, CH_BACKGR] = 0

    return (clean * 255).astype("uint8")


def perform_otsu(channel):
    return channel > skimage.filters.threshold_otsu(channel)


def circle(r: int) -> np.ndarray:
    """returns circular mask with a radius of r pixels"""
    dim = r * 2 + 1
    arr = np.zeros((dim, dim))
    for i in range(dim):
        dst_x = pow(r - i, 2)
        for j in range(dim):
            dst_y = pow(r - j, 2)
            if math.sqrt(dst_x + dst_y) <= (dim) / 2:
                arr[i, j] = 1
    return arr


def get_clean_tubules(seg: np.ndarray, min_tubule_size: int) -> np.ndarray:
    """
    removes small tubules from the image and fills holes in remaining ones
    Uses labelling to separate tubule and a min_tubule_size threshold to determine which tubules to keep
    """

    tubule = seg[:, :, CH_TUBULE]
    tubule = skimage.morphology.binary_opening(tubule, circle(6))
    tubule = skimage.morphology.binary_closing(tubule, circle(12))

    labeled_array, num_features = scipy.ndimage.label(tubule*255)
    groups, counts = np.unique(labeled_array, return_counts=True)
    groups = groups[counts > min_tubule_size]

    processed = labeled_array.copy()
    for i in range(num_features):
        object = labeled_array == i
        processed[object] = 0
        if i in groups:
            print(".", end="")
            processed[object] = i

    print("")
    return processed


def post_process(seg: np.ndarray, thresholding: Literal["argmax", "otsu"], level: int) -> np.ndarray:
    """
    performs post-processing
    """

    print("Performing post-processing...")
    clean = get_clean_image(seg, thresholding)
    print("got clean image")

    print("processing tubules")
    tubules = get_clean_tubules(clean, 6000 if level == 1 else 1500)

    print("\tdilating tubules. Might take a while")
    big_tub = binary_dilation(tubules, circle(100 if level == 1 else 25))

    #XXX do we want to remove pepper from borders?
    print("\tunpeppering borders")
    brd = skimage.morphology.binary_closing(clean[:, :, CH_BORDER], circle(3))

    #TODO: process border for every tubule separately

    # only allow borders that overlap enlarged tubule area
    print("\tgenerating clean borders")
    clean_border = np.logical_and(big_tub, brd)

    print("\tcleaning up tubules once more")
    tub = np.logical_and(tubules > 0, clean_border == False)
    # tub =  morphology.binary_dilation(tub,)*(1-clean_border)

    out = np.zeros_like(clean).astype('uint8')
    out[:, :, CH_BORDER] = (clean_border * 255).astype('uint8')
    out[:, :, CH_TUBULE] = (tub * 255).astype('uint8')
    out[:, :, CH_BACKGR] = (255 * np.logical_and(out[:, :, CH_TUBULE] == 0, out[:, :, CH_BORDER] == 0)).astype('uint8')

    return out



