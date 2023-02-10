# Script for generating image slices (step 1 of 3)
# Generates training image slices from a provided OpenSlide slide
# Can create test data, training data, or both
# Supports various tubule overlap settings and zoom levels
import argparse
import json
import shutil
import os
import statistics
from typing import List, Optional

import openslide
from dotenv import load_dotenv

from augmentations import apply_random_augmentation
from slide_utils import load_annotations, get_random_bounds, get_slice, get_annotated_area_ratio, print_progress, \
    AnnotationsGroup, get_slide_offset, Polygon
from os.path import abspath

def generate_slices(
        slide: openslide.OpenSlide,
        sampling_polygons: List[Polygon],
        annotations: AnnotationsGroup,
        output_folder: str,
        num_images: int,
        level: int,
        tubule_threshold: int,
        wiggle=0.85,
        width=512,
        height=512,
):
    print(f"\n\nLevel {level}, tubule_threshold {tubule_threshold}:")

    # prepare output dir
    if os.path.exists(output_folder):
        print(f"removing dir {output_folder}")
        shutil.rmtree(output_folder)
    print(f"Making dir {output_folder}")
    os.makedirs(output_folder)

    meta = {
        "slices": list()
    }
    img_idx = 0
    while img_idx < num_images:
        bounds = get_random_bounds(sampling_polygons, level, width, height, wiggle)
        # TODO: maybe generate bounds truly randomly, from region, not from annotations?
        pics = get_slice(slide, level, bounds, annotations, debug=True)
        if pics is None:
            print(f"Failed to get pic at bounds {bounds}")
            continue

        orig, seg, debug = pics


        tubule_ratio = get_annotated_area_ratio(seg)
        if tubule_ratio > tubule_threshold / 100:
            images, augmentations = apply_random_augmentation([orig, seg, debug], 2)
            orig, seg, debug = images

            meta["slices"].append({
                "id": img_idx,
                "tubule_ratio": round(tubule_ratio, 3),
                "bounds": bounds.__dict__,
                "augmentations": augmentations
            })

            orig.save(f"{output_folder}/slice{img_idx}_orig.png")
            seg.save(f"{output_folder}/slice{img_idx}_seg.png")
            debug.save(f"{output_folder}/slice{img_idx}_debug.png")

            img_idx += 1
            print_progress(img_idx, 100, num_images)

    mean_ratio = statistics.mean([sliceinfo["tubule_ratio"] for sliceinfo in meta["slices"]])
    meta["mean_ratio"] = round(mean_ratio, 3)
    print(f"Mean ratio: {round(mean_ratio, 3)}")

    with open(f"{output_folder}/meta.json", "w") as fp:
        json.dump(meta, fp, indent=2)


if __name__ == "__main__":
    load_dotenv()

    class Args(argparse.Namespace):
        slidefile: str
        annotationsfile: str
        level: int
        tubule_threshold: int

        num_train_images: int
        num_test_images: int
        train_test_split: int

        folder_prefix: Optional[str]

    parser = argparse.ArgumentParser()
    parser.add_argument("--slidefile", type=str, required=True)
    parser.add_argument("--annotationsfile", type=str, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--tubule_threshold", type=int, required=True)

    parser.add_argument("--num_train_images", type=int, default=0)
    parser.add_argument("--num_test_images", type=int, default=0)
    parser.add_argument("--train_test_split", type=int, default=0)
    parser.add_argument("--folder_prefix", type=str)

    args: Args = parser.parse_args()

    pieces = args.slidefile.split("/")
    data_folder = "/".join(pieces[:-1])


    print(f"Opening {abspath(os.path.expanduser(args.slidefile))}")
    slide = openslide.open_slide(abspath(os.path.expanduser(args.slidefile)))
    print(f"Opened slide. Dims={slide.level_dimensions[0]}, Offset={get_slide_offset(slide)}")

    annotations = load_annotations(args.annotationsfile, get_slide_offset(slide))
    print(f"Loaded annotations: outsides={len(annotations.outsides)}, insides={len(annotations.insides)}")

    modes = []
    if args.num_train_images > 0:
        modes.append("train")
    if args.num_test_images > 0:
        modes.append("test")


    for purpose in modes:
        if purpose == "train":
            sampling = annotations.outsides[args.train_test_split:]
            num_images = args.num_train_images

        elif purpose == "test":
            if args.train_test_split > 0:
                sampling = annotations.outsides[:args.train_test_split]
            else:
                sampling = annotations.outsides

            num_images = args.num_test_images

        else:
            print("no purpose")
            continue


        slide_name = args.slidefile.split("/")[-1]
        prefix = f"{args.folder_prefix}/" if args.folder_prefix is not None else ""

        generate_slices(
            slide=slide,
            sampling_polygons=sampling,
            annotations=annotations,
            output_folder=f"{os.environ.get('slides')}/{prefix}{slide_name}/level{args.level}_overlap{args.tubule_threshold}/{purpose}",
            num_images=num_images,
            level=args.level,
            tubule_threshold=args.tubule_threshold,
        )

