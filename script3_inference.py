# Script for inference (step 3 of 3)

# takes a trained model and pre-generated mosaic slides
# (inference mode, regular slides not random)
# predicts the output from the slides, saves them and
# constructs a large output image (with and without borders)

# It's the main output of the project once it works well

import tensorflow as tf

from settings import load_env

if tf.test.is_gpu_available():
    print("GPU is available, ready to go")
else:
    raise Exception("GPU not found. Cannot continue")

from typing import List, Literal, Tuple
from PIL import ImageDraw, Image
import numpy as np
import os
import glob
from segmentation_models import Unet
from segmentation_models.losses import categorical_crossentropy
from segmentation_models.metrics import iou_score, f1_score
from functools import cmp_to_key
import shutil


def load_image(filename: str, color_mode: Literal["color", "grayscale"]) -> np.ndarray:
    img = Image.open(filename)
    if color_mode == "grayscale":
        img = img.convert('L')
        img = np.array(img)
    else:
        img = np.array(img)
        img = img[..., 0:-1] # remove alpha layer

    return img


def load_mosaic(level: int, colmode: Literal["color", "grayscale"]) -> Tuple[List[np.ndarray], List[str]]:
    """
    loads all mosaic images in a folder to an unsorted 1d-array.
    """
    os.environ['workdir']
    folder = f"{os.environ['workdir']}/inference/lvl{level}"
    path_length = len(folder)+1
    pathnames: List[str] = glob.glob(f"{folder}/row*_col*_orig.png")
    filenames = [f[path_length:] for f in pathnames]

    images = [load_image(f"{folder}/{f}", colmode) for f in filenames]

    return images, filenames


def join_pieces(pieces: List[List[Image.Image]], overlap=0.25, show_borders=False) -> Image:
    """
    Joins 2d-array of images into one large image.
    """
    piece_width = pieces[0][0].width  # 512 px
    piece_height = pieces[0][0].height # 512 px

    result_width = int(piece_width * (1-overlap) * len(pieces[0]))
    result_height = int(piece_height * (1-overlap) * len(pieces))

    output = Image.new("RGB", (result_width, result_height))

    for row in range(len(pieces)):
        for col in range(len(pieces[row])):
            img = pieces[row][col]
            region = img.crop((
                (int(piece_width * overlap / 2)),
                (int(piece_height * overlap / 2)),
                (int(piece_width * (1 - overlap / 2))),
                (int(piece_height * (1 - overlap / 2)))
            ))

            if show_borders:
                drw = ImageDraw.Draw(region)
                drw.rectangle((0, 0, int(piece_width * (1 - overlap / 2)), int(piece_width * (1 - overlap / 2))), outline="black", width=1)

            pos = (
                int(col * piece_width * (1-overlap)),
                int(row * piece_height * (1-overlap))
            )

            output.paste(region, pos)

    return output


def create_model(color_mode, backbone="resnet34"):
    num_channels = 1 if color_mode=="grayscale" else 3
    model = Unet(backbone, encoder_weights=None, input_shape=(512, 512, num_channels), classes=3)
    model.compile('Adam', loss=categorical_crossentropy, metrics=[iou_score, f1_score])
    return model


def get_row_col(filename):
    bits = filename.split("_")
    return (int(bits[0][3:]), int(bits[1][3:]))


def mosaic_comparator(a, b):
    if a[:3] == "row" and b[:3] == "row":
        first = get_row_col(a)
        second = get_row_col(b)
        if first[0]==second[0]:
            return first[1]-second[1]
        return first[0]-second[0]
    raise Exception(f"bad filenames: {(a, b)}")


def generate_outcomes(filenames: List[str], images: List[np.ndarray], level: int, color_mode: str, overlap: int):
    print(f"\nGenerating outcomes: {level}, {color_mode}, {overlap}. ({len(filenames)} images) \n")

    if len(images) != len(filenames):
        raise Exception(f"Nr filenames and images are different: {len(filenames)}, {len(images)}")


    print("\nPredicting segmentation slices\n")
    model = create_model(color_mode)
    os.environ.get('models')
    model.load_weights(f"{os.environ['models']}/{os.environ['modelname']}/checkpoint.ckpt")
    preds = model.predict(np.array(images))


    target = f"output/lvl{level}_overlap{overlap}_col{color_mode}_out"
    print(f"\nSaving segmentation pieces to dir '{target}'\n")

    if os.path.exists(target):
        print("Target dir exists, removing it")
        shutil.rmtree(target)

    os.makedirs(target)

    for i in range(len(preds)):
        img = Image.fromarray((preds[i]*255).astype('uint8'))
        fname = filenames[i].replace("orig", "seg")
        img.save(f"{target}/{fname}")
        print(".", end="")
        if (i + 1) % 100 == 0:
            print("")


    # load segmentation pieces into 2d-array
    print("\nLoading segmentation pieces\n")
    sorted_pics = sorted(filenames, key=cmp_to_key(mosaic_comparator))
    seg2d = []
    last_row = -1
    for fname in sorted_pics:
        row, col = get_row_col(fname)
        if row > last_row:
            last_row = row
            seg2d.append([])

        arr = seg2d[row]
        fname = fname.replace("orig", "seg")
        arr.append(Image.open(f"{target}/{fname}"))


    # finally. generate mosaic images
    basename = f"output/lvl{level}_overlap{overlap}_col{color_mode}.composite"
    print(f"Generating mosaic images: {basename}")

    out_borders = join_pieces(seg2d, show_borders=1)
    out_borders.save(f"{basename}.borders.png")
    out_borders.thumbnail([2000, 2000])
    out_borders.save(f"{basename}.borders.THUMBNAIL.png")

    out_clean = join_pieces(seg2d)
    out_clean.save(f"{basename}.clean.png")
    out_clean.thumbnail([2000, 2000])
    out_clean.save(f"{basename}.clean.THUMBNAIL.png")

    print("\n\nFinished output generation\n\n")


IDX_IMAGES = 0
IDX_FILENAMES = 1


if __name__ == "__main__":
    load_env()

    print("Loading source images")
    images_1_color = load_mosaic(1, "color")
    images_1_gray = load_mosaic(1, "grayscale")
    # images_2_color = load_mosaic(2, "color")
    # images_2_gray = load_mosaic(2, "grayscale")

    generate_outcomes(filenames=images_1_color[IDX_FILENAMES], images=images_1_color[IDX_IMAGES], level=1, color_mode="color", overlap=10)
    generate_outcomes(filenames=images_1_gray[IDX_FILENAMES], images=images_1_gray[IDX_IMAGES], level=1, color_mode="grayscale", overlap=10)

    # generate_outcomes(filenames=images_2_color[IDX_FILENAMES], images=images_2_color[IDX_IMAGES], level=2, color_mode="color", overlap=50)
    # generate_outcomes(filenames=images_2_color[IDX_FILENAMES], images=images_2_color[IDX_IMAGES], level=2, color_mode="color", overlap=30)
    # generate_outcomes(filenames=images_2_color[IDX_FILENAMES], images=images_2_color[IDX_IMAGES], level=2, color_mode="color", overlap=10)
    #
    # generate_outcomes(filenames=images_2_gray[IDX_FILENAMES], images=images_2_gray[IDX_IMAGES], level=2, color_mode="grayscale", overlap=50)
    # generate_outcomes(filenames=images_2_gray[IDX_FILENAMES], images=images_2_gray[IDX_IMAGES], level=2, color_mode="grayscale", overlap=30)
    # generate_outcomes(filenames=images_2_gray[IDX_FILENAMES], images=images_2_gray[IDX_IMAGES], level=2, color_mode="grayscale", overlap=10)

    print("Script finished")
