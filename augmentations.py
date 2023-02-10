# contains image augmentation functions

import random
from typing import List, Tuple

from PIL import Image, ImageOps
from PIL.Image import Transpose


def rotate90(image: Image.Image):
    return image.transpose(Transpose.ROTATE_90)


def rotate180(image: Image.Image):
    return image.transpose(Transpose.ROTATE_180)


def rotate270(image: Image.Image):
    return image.transpose(Transpose.ROTATE_270)


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
