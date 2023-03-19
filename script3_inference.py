# Script for inference (step 3 of 3)
from openslide import OpenSlide

# takes a trained model and pre-generated mosaic slides
# (inference mode, regular slides not random)
# predicts the output from the slides, saves them and
# constructs a large output image (with and without borders)

# It's the main output of the project once it works well

from settings import load_env
from slide_utils import load_annotations, Bounds, X_COORD, Y_COORD
from utils.model import create_model

from typing import List, Literal, Tuple
from PIL import ImageDraw, Image
import numpy as np
import os
import glob
import shutil


def load_image(filename: str, color_mode: Literal["color", "grayscale"]) -> np.ndarray:
    img = Image.open(filename)
    if color_mode == "grayscale":
        img = img.convert('L')
        img = np.array(img)
    else:
        img = np.array(img)
        img = img[..., 0:-1] # remove alpha layer

    return img / 255.0


def load_mosaic_images(slide:str, level: int, overlap: int, color_mode: Literal["color", "grayscale"]) -> Tuple[List[np.ndarray], List[str]]:
    """
    loads all mosaic images in a folder to an unsorted 1d-array.
    """
    folder = f"{os.environ['workdir']}/inference/{slide}/lvl{level}_overlap{overlap}"
    path_length = len(folder)+1
    pathnames: List[str] = glob.glob(f"{folder}/row*_col*_orig.png")
    filenames = [f[path_length:] for f in pathnames]

    print(f"Loading {len(filenames)} images")
    images = [load_image(f"{folder}/{f}", color_mode) for f in filenames]
    print("done")
    return images, filenames


def save_output_images(target_dir: str, images: List[np.ndarray], filenames: List[str]):
    """
    Saves list of images (in ndarray form) to disk, using filenames as a naming guide (replaces _orig with _seg)
    Image values are supposed to be normalized (in 0...1 range)
    """
    for i in range(len(images)):
        img = Image.fromarray((images[i] * 255).astype('uint8'))
        fname = filenames[i].replace("orig", "seg")
        img.save(f"{target_dir}/{fname}")
        print(".", end="")
        if (i + 1) % 100 == 0:
            print("")
    print("")


def get_row_id(filename: str) -> int:
    return int(filename.split("_")[0][3:])


def get_col_id(filename):
    return int(filename.split("_")[1][3:])


def build_mosaic_2d_array(images: List[np.ndarray], filenames: List[str]) -> List[List[Image.Image]]:

    # sort images based on custom comparator in filenames (e.g. row7_col9_orig.png). row by row, col by col
    dtype = [('filename', str, 32), ('row', int), ('col', int)]
    files = np.array(
        [(f, get_row_id(f), get_col_id(f)) for f in filenames],
        dtype=dtype
    )
    indices = np.argsort(files, order=["row","col"])

    arr2d: List[List[Image.Image]] = []
    last_row = -1
    for idx in indices:
        filename, row, col = files[idx]

        if row > last_row:
            last_row = row
            arr2d.append([])

        img = Image.fromarray((images[idx]*255).astype('uint8'))
        arr2d[row].append(img)

    return arr2d


def get_slices(slide: OpenSlide, bounds: Bounds, level: int, overlap: float, slice_size=512):
    zoom = int(pow(2, level))

    # 512*0.50 / 2 = 256/2 = 128
    # 512*0.25 / 2 = 128/2 =  64  (smaller buffer area since we're using more of the initial image
    padding = int(slice_size * overlap / 2)
    final_padding = zoom * padding

    topleft_corner_slide_coords = (
        bounds.topleft[X_COORD] - final_padding,
        bounds.topleft[Y_COORD] - final_padding,
    )

    tubule_area_width, tubule_area_height = (
        bounds.get_width() + final_padding * 2,
        bounds.get_height() + final_padding * 2,
    )

    pieces = []
    shift = slice_size * (1 - overlap) * zoom
    # sliceify
    slice_x, slice_y = 0, 0
    while slice_y < tubule_area_height:
        slice_x = 0
        row = []
        while slice_x < tubule_area_width:
            tl = (
                topleft_corner_slide_coords[X_COORD] + slice_x,
                topleft_corner_slide_coords[Y_COORD] + slice_y,
            )
            img = slide.read_region(location=tl, level=level, size=(slice_size, slice_size))
            row.append(img)
            slice_x = int(slice_x + shift )

        pieces.append(row)
        slice_y = int(slice_y + shift)

    return pieces


def join_pieces(pieces: List[List[Image.Image]], overlap=0.25, show_borders=False, expected_size: Tuple[int, int] = None) -> Image.Image:
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

    if expected_size is not None:
        # inference region might not map to slices precisely. cut down excess
        # set right and bottom to desired size

        if expected_size[0] > result_width or expected_size[1] > result_height:
            raise Exception(f"Expected size larger than slide area! {expected_size} vs {(result_width, result_height)}")

        return output.crop((
            0,
            0,
            expected_size[0],
            expected_size[1]
        ))

    return output


def prepare_directory(path: str):
    print(f"Output directory: {path}")

    if os.path.exists(path):
        print("Output directory exists, removing it")
        shutil.rmtree(path)

    os.makedirs(path)


def get_area_size(slidename, region, level):

    annotations = load_annotations(os.environ.get(f'annotations_{slidename}'), (0,0))
    region = annotations.regions[region]

    bounds = Bounds(
        topleft=region[0],
        topright=region[1],
        bottomleft=region[3],
        bottomright=region[2],
        zoom_level=level,
    )

    return bounds.get_size()


def generate_outcomes(
        filenames: List[str],
        images: List[np.ndarray],
        slidename: str,
        overlap: float,
        color_mode: str,
        model_name: str,
        area_size: (int, int),
        save_pieces: bool,
        save_composite: bool,
):
    print(f"\nGenerating outcomes:  {color_mode}. ({len(filenames)} images) \n")
    if len(images) != len(filenames):
        raise Exception(f"Nr filenames and images are different: {len(filenames)}, {len(images)}")

    print("\nPredicting segmentation slices\n")
    model = create_model(color_mode)
    model.load_weights(f"{os.environ['models']}/{model_name}/checkpoint.ckpt")

    mid = int(len(images)/2)
    img1 = np.array(images[:mid])
    img2 = np.array(images[mid:])
    preds1 = model.predict(img1)
    preds2 = model.predict(img2)
    preds = np.concatenate([preds1, preds2])
    if save_pieces:
        print("Saving segmentation slices")
        target = f"output/{model_name}/pieces.{slidename}.{round(overlap*100)}"
        prepare_directory(target)
        save_output_images(target, preds, filenames)

    #Generate mosaic images
    seg2d = build_mosaic_2d_array(preds, filenames)
    basename = f"output/{model_name}/composite.{slidename}.{round(overlap*100)}"
    print(f"Generating mosaic images: {basename}")

    out_borders = join_pieces(seg2d, overlap=overlap, show_borders=True, expected_size=area_size)
    out_borders_thumb = out_borders.copy()
    out_borders_thumb.thumbnail([2000, 2000])

    out_clean = join_pieces(seg2d, overlap=overlap, expected_size=area_size)
    out_clean_thumb = out_clean.copy()
    out_clean_thumb.thumbnail([2000, 2000])

    if save_composite:
        out_borders.save(f"{basename}.borders.png")
        out_borders_thumb.save(f"{basename}.borders.THUMBNAIL.png")
        out_clean.save(f"{basename}.clean.png")
        out_clean_thumb.save(f"{basename}.clean.THUMBNAIL.png")

    print("\n\nFinished output generation\n\n")
    return preds, out_borders_thumb, out_clean_thumb


IDX_IMAGES = 0
IDX_FILENAMES = 1


if __name__ == "__main__":
    load_env()

    print("Loading source images")
    images_1_color = load_mosaic_images(os.environ['slidename'], 1, int(os.environ['mosaic_overlap']), "color")

    generate_outcomes(
        filenames=images_1_color[IDX_FILENAMES],
        images=images_1_color[IDX_IMAGES],
        slidename=os.environ['slidename'],
        overlap=int(os.environ['mosaic_overlap'])/100,
        color_mode="color",
        model_name=os.environ['model_name'],
        save_pieces=True,
        save_composite=True,
    )

    # images_1_gray = load_mosaic(1, "grayscale")
    # images_2_color = load_mosaic(2, "color")
    # images_2_gray = load_mosaic(2, "grayscale")
    # generate_outcomes(filenames=images_1_gray[IDX_FILENAMES], images=images_1_gray[IDX_IMAGES], level=1, color_mode="grayscale", overlap=10)

    # generate_outcomes(filenames=images_2_color[IDX_FILENAMES], images=images_2_color[IDX_IMAGES], level=2, color_mode="color", overlap=50)
    # generate_outcomes(filenames=images_2_color[IDX_FILENAMES], images=images_2_color[IDX_IMAGES], level=2, color_mode="color", overlap=30)
    # generate_outcomes(filenames=images_2_color[IDX_FILENAMES], images=images_2_color[IDX_IMAGES], level=2, color_mode="color", overlap=10)
    #
    # generate_outcomes(filenames=images_2_gray[IDX_FILENAMES], images=images_2_gray[IDX_IMAGES], level=2, color_mode="grayscale", overlap=50)
    # generate_outcomes(filenames=images_2_gray[IDX_FILENAMES], images=images_2_gray[IDX_IMAGES], level=2, color_mode="grayscale", overlap=30)
    # generate_outcomes(filenames=images_2_gray[IDX_FILENAMES], images=images_2_gray[IDX_IMAGES], level=2, color_mode="grayscale", overlap=10)

    print("Script finished")
