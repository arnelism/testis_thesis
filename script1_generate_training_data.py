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

from augmentations import apply_random_augmentation
from settings import load_env
from slide_utils import load_annotations, get_random_bounds, get_slice, get_annotated_area_ratio, print_progress, \
    AnnotationsGroup, get_slide_offset, Polygon
from os.path import abspath

def generate_slices(
        slide: openslide.OpenSlide,
        sampling_polygons: List[Polygon],
        annotations: AnnotationsGroup,
        region_handling: str,
        output_folder: str,
        num_images: int,
        level: int,
        tubule_threshold: int,
        wiggle=0.85,
        width=512,
        height=512,
):
    print(f"\n\nLevel {level}, tubule_threshold {tubule_threshold}:")
    print(f"Test regions: {annotations.regions}")
    print(list(annotations.regions.values()))
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
        if region_handling == "contain":  # test data
            bounds = get_random_bounds(sampling_polygons, level, width, height, wiggle, contain=list(annotations.regions.values()))
        elif region_handling == "exclude":  # train data
            bounds = get_random_bounds(sampling_polygons, level, width, height, wiggle, exclude=list(annotations.regions.values()))
        else:
            raise Exception(f"Missing/invalid value for region_handling ('{region_handling}')")

        if bounds is None:
            raise Exception("Missing Bounds")

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
    load_env()

    slide_name = os.environ['slide_name']
    slidefile = os.environ[f"slidefile_{slide_name}"]
    annotationsfile = os.environ[f"annotations_{slide_name}"]
    test_regions = os.environ.get("test_regions")

    level = int(os.environ["level"])
    tubule_threshold = int(os.environ["tubule_threshold"])
    num_train_images = int(os.environ["num_train_images"])
    num_test_images = int(os.environ["num_test_images"])

    folder_prefix = os.environ.get("folder_prefix")

    print(f"Opening {abspath(os.path.expanduser(f'{slidefile}'))}")
    slide = openslide.open_slide(abspath(os.path.expanduser(slidefile)))
    print(f"Opened slide. Dims={slide.level_dimensions[0]}, Offset={get_slide_offset(slide)}")

    annotations = load_annotations(annotationsfile, get_slide_offset(slide))
    if test_regions != "all" and test_regions is not None:
        annotations.regions = {region_name.strip(): annotations.regions[region_name.strip()] for region_name in test_regions.split(",")}

    print(f"Loaded annotations: outsides={len(annotations.outsides)}, insides={len(annotations.insides)}")
    print(f"Using regions: {annotations.regions.keys()}")

    modes = []
    if num_train_images > 0:
        modes.append("train")
    if num_test_images > 0:
        modes.append("test")

    for purpose in modes:
        if purpose == "train":
            num_images = num_train_images

        elif purpose == "test":
            num_images = num_test_images

        else:
            print("no purpose")
            continue


        # slide_name = args.slidefile.split("/")[-1]
        prefix = f"{folder_prefix}/" if folder_prefix is not None else ""

        generate_slices(
            slide=slide,
            sampling_polygons=annotations.outsides,
            annotations=annotations,
            output_folder=f"{os.environ.get('slides')}/{prefix}{slide_name}/level{level}_overlap{tubule_threshold}/{purpose}",
            num_images=num_images,
            level=level,
            tubule_threshold=tubule_threshold,
            region_handling="contain" if purpose == "test" else "exclude"
        )

