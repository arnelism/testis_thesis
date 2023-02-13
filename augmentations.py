# contains image augmentation functions

import random
from typing import List, Tuple

from PIL import Image, ImageOps

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
ROTATE_90 = 2
ROTATE_180 = 3
ROTATE_270 = 4
TRANSPOSE = 5
TRANSVERSE = 6


def rotate90(image: Image.Image):
    return image.transpose(ROTATE_90)


def rotate180(image: Image.Image):
    return image.transpose(ROTATE_180)


def rotate270(image: Image.Image):
    return image.transpose(ROTATE_270)


def noop(image: Image.Image):
    return image


AUGMENTATIONS = [
    noop,
    noop,
    noop,
    noop,
    ImageOps.flip,
    ImageOps.mirror,
    rotate90,
    rotate180,
    rotate270,
]


def apply_random_augmentation(images: List[Image.Image], rounds=1) -> Tuple[List[Image.Image], List[str]]:
    # make copy to avoid modifying input data
    images = [img for img in images]
    augmentations = []
    for _ in range(rounds):
        aug = random.choice(AUGMENTATIONS)
        augmentations.append(aug.__qualname__)
        for i in range(len(images)):
            images[i] = aug(images[i])

    return images, augmentations
