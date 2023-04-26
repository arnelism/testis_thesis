from typing import List

import numpy as np
import skimage

from image_utils import CH_BACKGR, CH_TUBULE


def get_edges(image: np.ndarray, channel: int) -> np.ndarray:
    binarized = image[:, :, channel] > 0
    return skimage.morphology.binary_dilation(binarized, footprint=np.ones((3, 3))) != binarized


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def calc_wall_thickness(seg: np.ndarray, accuracy: int):
    """
    Calculates wall thicknessesses on the segmentation map (shaped W x H x 3)
    Seg should be post-processed
    For every pixel on tubules' borders inner edge, sets a value which is a distance to the outer edge
    Every other pixel set to 0
    Results should be interpreted by analyzing pixels that are > 0
    """
    max_range = 400
    masks = [create_circular_mask(i, i) for i in range(max_range * 2)]

    # find tubule and background edge lines
    bg_edges = get_edges(seg, CH_BACKGR)
    tubule_edges = get_edges(seg, CH_TUBULE)

    # pad them with maximum border width so that we can apply the algorithm
    bg_edges_padded = np.pad(bg_edges, max_range)
    tubule_edges_padded = np.pad(tubule_edges, max_range )

    #initialize empty result img
    width_map = np.zeros_like(tubule_edges_padded).astype('uint')

    # for each point on tubule_edge line, find distance to the nearest point on background-edge line
    for row in range(max_range, tubule_edges_padded.shape[0]-max_range):
        if row % 100 == 0:
            print(row)  # primitive progress bar

        for col in range(max_range, tubule_edges.shape[1]-max_range):

            if tubule_edges_padded[row, col]:  # px is on the tubule edge line, calculate distance to background edge

                # fit a circle centered at the inner line. how large circle fits? increment by ACCURACY
                for i in range(0, max_range, accuracy):
                    # take a surrounding region from bg edges, radius=i
                    top = row - i
                    bottom = row + i + 1
                    left = col - i
                    right = col + i + 1
                    region = bg_edges_padded[top:bottom, left:right]

                    # make region circular
                    mask = masks[i * 2 + 1]
                    if region.shape[1] != mask.shape[1]:
                        print(row, col, i, mask.shape, region.shape)
                    else:
                        region = np.logical_and(region, mask)

                    # finally, if there are any line-pixels within range, report i as distance
                    if np.any(region):
                        width_map[row, col] = i
                        break

    return width_map


def get_width_percentiles(width_map: np.ndarray) -> List[float]:
    return [
        np.percentile(width_map[width_map > 0], 5),
        np.percentile(width_map[width_map > 0], 25),
        np.percentile(width_map[width_map > 0], 50),
        np.percentile(width_map[width_map > 0], 75),
        np.percentile(width_map[width_map > 0], 95),
        np.percentile(width_map[width_map > 0], 99),
        np.percentile(width_map[width_map > 0], 99.9),
    ]
