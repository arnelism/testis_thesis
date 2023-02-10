# Script for generating image slices (step 1 of 3)
# Generates training image slices from a provided OpenSlide slide
# Can create test data, training data, or both
# Supports various tubule overlap settings and zoom levels
import json
import shutil
import os
import statistics
from typing import List

import openslide

from augmentations import apply_random_augmentation
from slide_utils import load_annotations, get_random_bounds, get_slice, get_annotated_area_ratio, print_progress, \
    AnnotationsGroup, get_slide_offset, Polygon


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
        orig, seg, debug = get_slice(slide, level, bounds, annotations, debug=True)


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


# TODO pass data folder via env
data_folder = "/Users/arnel/repos/thesisData/dataset"
slidefile = f"{data_folder}/19,H,16747,_,01,1,0.mrxs"
slide = openslide.open_slide(slidefile)
print(f"Opened slide. Dims={slide.level_dimensions[0]}, Offset={get_slide_offset(slide)}")

# TODO pass via env or cli params
annotations = load_annotations("annotations/checkpoint.2023-01-31_2258.geojson", get_slide_offset(slide))
print(f"Loaded annotations: outsides={len(annotations.outsides)}, insides={len(annotations.insides)}")


# output_folder = f"data/slides/level{level}_overlap{tubule_threshold}/{purpose}"
level = 1
for purpose in ["train", "test"]:
    if purpose == "train":
        sampling = annotations.outsides[5:]
    else:
        sampling = annotations.outsides[:5]

    generate_slices(
        slide=slide,
        sampling_polygons=sampling,
        annotations=annotations,
        output_folder=f"data/test/level{level}_overlap60/{purpose}",
        num_images=10,
        level=level,
        tubule_threshold=10,
    )

# width = 512
# height = 512
# wiggle = 0.8
# NUM_IMAGES_TRAIN = 10
# NUM_IMAGES_TEST = 5
#
#
#

#
# level = 1
# # transform annotations
# annotations = load_annotations("./checkpoint20230123_0116.geojson", get_offset(slide))
# print(f"Loaded annotations: {len(annotations.outsides)}, {len(annotations.insides)}")
