# Helper functions for working with slides and annotations
from __future__ import annotations
from dataclasses import dataclass
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import openslide
import json

from PIL import Image, ImageDraw
import shapely

def get_slide_offset(slide: openslide.OpenSlide) -> Tuple[int, int]:
    return (
        int(slide.properties['openslide.bounds-x']),
        int(slide.properties['openslide.bounds-y'])
    )


X_COORD = 0
Y_COORD = 1

Coordinate = Tuple[int, int]
Polygon = List[Coordinate]


@dataclass
class AnnotationsGroup:
    outsides: List[Polygon]
    insides: List[Polygon]
    regions: Dict[str, Polygon]


@dataclass
class Bounds:
    topleft: Coordinate
    topright: Coordinate
    bottomleft: Coordinate
    bottomright: Coordinate
    # xxx zoom level is stored here because some bounds calculations depend on it and one cannot provide it at calltime
    zoom_level: int

    def get_width(self) -> int:
        """
        Get width of the bounds in zoom level 0
        """
        return self.topright[X_COORD] - self.topleft[X_COORD]

    def get_height(self) -> int:
        """
        Get height of the bounds in zoom level 0
        """
        return self.bottomleft[Y_COORD] - self.topleft[Y_COORD]

    def get_size(self) -> (int, int):
        """
        Get width and height of the bounds in configured zoom level
        """
        scale = pow(2, self.zoom_level)
        return int(self.get_width() / scale), int(self.get_height() / scale)

    def overlaps(self, another: Bounds) -> bool:
        """
        Whether or not two Bounds overlap (should convert to shapely)
        """
        if another.topleft[X_COORD] <= self.topright[X_COORD] <= another.topright[X_COORD]:
            if another.topleft[Y_COORD] <= self.bottomleft[Y_COORD] <= another.bottomleft[Y_COORD]:
                return True

        return False

    def to_polygon(self) -> Polygon:
        """
        Convert to polygon (array or coordinates) denoting corners of the bounds
        """
        return [self.topleft, self.topright, self.bottomright, self.bottomleft]


def load_annotations(filename: str, offset: Tuple[int, int], is_feature_collection=True) -> AnnotationsGroup:
    with open(filename, "r") as fp:
        ann = json.load(fp)

    features = ann["features"] if is_feature_collection else ann

    outsides = list(filter(lambda f: f["properties"]["classification"]["name"] == "tubule outside", features))
    insides = list(filter(lambda f: f["properties"]["classification"]["name"] == "tubule inside", features))
    regions = list(filter(lambda f: f["properties"]["classification"]["name"] == "test region", features))

    print(f"Applying offset {offset}")
    return AnnotationsGroup(
        outsides=[apply_offset(get_annotation(ann), offset) for ann in outsides],
        insides=[apply_offset(get_annotation(ann), offset) for ann in insides],
        regions={ann["properties"]["name"]: apply_offset(get_annotation(ann), offset) for ann in regions},
    )


def get_annotation(ann: dict) -> Polygon:
    if ann["geometry"]["type"] == "Polygon":
        coords: List[Tuple[int, int]] = ann["geometry"]["coordinates"][0]
    elif ann["geometry"]["type"] == "MultiPolygon":
        print(f"Warning: feature {ann.get('id')} is multiPolygon")
        coords = list()
        for candidate in ann["geometry"]["coordinates"]:
            if len(candidate[0]) > len(coords):
                coords: List[Tuple[int, int]] = candidate[0]

    return [( round(c[X_COORD]), round(c[Y_COORD])) for c in coords]


def apply_offset(ann: Polygon, offset: Tuple[int, int]) -> Polygon:
    if len(ann) == 1:
        ann = ann[0]
    answer = [(c[X_COORD] + offset[X_COORD], c[Y_COORD] + offset[Y_COORD]) for c in ann]
    return answer


def get_random_bounds(
        annotations: List[Polygon],
        level: int,
        width: int,
        height: int,
        wiggle: float,
        contain: Optional[List[Polygon]] = None,
        exclude: Optional[List[Polygon]] = None,
        attempts: int = 1000
) -> Bounds:
    for i in range (attempts):
        # pick random annotation
        poly = annotations[random.randint(0, len(annotations) - 1)]

        # get a random point of the polygon boundary (use random vertice)
        poly_pt = poly[random.randint(0, len(poly) - 1)]

        # convert desired output dims to base (level0) image dims
        # The larger the zoom level (the more image area is covered)
        zoom = pow(2, level)
        source_width = width * zoom
        source_height = height * zoom

        # place the random point in the middle of the slide
        x = poly_pt[X_COORD] - int(source_width / 2)
        y = poly_pt[Y_COORD] - int(source_height / 2)

        # move the slide around so that it's not always in the dead centre
        wiggle_room_x = int(source_width * wiggle)
        wiggle_room_y = int(source_height * wiggle)
        x = x + random.randint(-wiggle_room_x, wiggle_room_x)
        y = y + random.randint(-wiggle_room_y, wiggle_room_y)

        candidate = Bounds(
            topleft=(x, y),
            topright=(x + source_width, y),
            bottomleft=(x, y + source_height),
            bottomright=(x + source_width, y + source_height),
            zoom_level=level,
        )

        if contain is not None:
            if not is_contained_in_any(candidate, contain):
                continue

        if exclude is not None:
            if intercepts_any(candidate, exclude):
                continue

        return candidate

    print("out of tries")
    return None


def get_polygon_bounds(polygon: Polygon, zoom: int) -> Bounds:
    """
    Returns bounding box of a polygon (polygon precisely fits into that box)
    """
    min_x = min([x for (x,y) in polygon])
    min_y = min([y for (x,y) in polygon])
    max_x = max([x for (x,y) in polygon])
    max_y = max([y for (x,y) in polygon])

    return Bounds(
        topleft=(min_x, min_y),
        topright=(max_x, min_y),
        bottomleft=(min_x, max_y),
        bottomright=(max_x, max_y),
        zoom_level=zoom
    )


def get_union_bounds(polygons: List[Polygon], zoom: int) -> Bounds:
    """
    Returns a bounding box containing all the polygons (full annotated area of a slide)
    """
    bounds_list = [get_polygon_bounds(p, zoom) for p in polygons]
    min_x = min([b.topleft[X_COORD] for b in bounds_list])
    min_y = min([b.topleft[Y_COORD] for b in bounds_list])
    max_x = max([b.bottomright[X_COORD] for b in bounds_list])
    max_y = max([b.bottomright[Y_COORD] for b in bounds_list])

    return Bounds(
        topleft=(min_x, min_y),
        topright=(max_x, min_y),
        bottomleft=(min_x, max_y),
        bottomright=(max_x, max_y),
        zoom_level=zoom
    )


def is_contained_in_any(bounds: Bounds, polygons: List[Polygon]):
    test_shape = shapely.Polygon(bounds.to_polygon())
    for poly in polygons:
        region = shapely.Polygon(poly)
        if region.contains(test_shape):
            return True

    return False


def intercepts_any(bounds: Bounds, polygons: List[Polygon]):
    test_shape = shapely.Polygon(bounds.to_polygon())
    for poly in polygons:
        region = shapely.Polygon(poly)
        if region.intersects(test_shape):
            return True

    return False


def get_slice(slide: openslide.OpenSlide, level: int, bounds: Bounds, annotations: AnnotationsGroup, debug=False):
    # img_pure = slide.read_region(location=bounds.topleft, level=level, size=bounds.get_size())
    try:
        img_pure = slide.read_region(location=bounds.topleft, level=level, size=bounds.get_size())
    except Exception as e:
        print("\nBAD BOUNDS:")
        print(bounds)
        print(e)
        return None
    img_seg = Image.new("RGBA", bounds.get_size())

    # go thru all polygons. for each draw it
    # TODO: maybe it's faster to determine overlap and only then draw?
    localized_outsides = [get_localized_polygon(bounds, poly) for poly in annotations.outsides]
    localized_insides = [get_localized_polygon(bounds, poly) for poly in annotations.insides]

    draw_polygons(img_seg, localized_outsides, fill="red")
    draw_polygons(img_seg, localized_insides, fill="green")

    if debug:
        img_debug = slide.read_region(location=bounds.topleft, level=level, size=bounds.get_size())
        draw_polygons(img_debug, localized_outsides, outline="red", width=4)
        draw_polygons(img_debug, localized_insides, outline="green", width=2)
        return img_pure, img_seg, img_debug
    else:
        return img_pure, img_seg


def get_localized_polygon(bounds: Bounds, polygon: Polygon) -> Polygon:
    """
    translate polygon coordinates to relative to bounds
    (polygon point that located at the same place as topleft corner of bounds should get coordinates (0,0)
    """
    # TODO: filter out polygons that fall completely outside the bounds
    zoom = pow(2, bounds.zoom_level)
    return [(int((x - bounds.topleft[X_COORD]) / zoom), int((y - bounds.topleft[Y_COORD]) / zoom)) for (x, y) in polygon]


def draw_polygons(img: Image.Image, polygons: List[Polygon], fill=None, outline=None, width=1):
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        # TODO: determine if we need to draw at all (if polygon is completely outside the bounds, skip drawing)
        draw.polygon(poly, fill, outline, width)


# TODO: try to calculate from bounds and polygons. maybe faster that way
def get_annotated_area_ratio(segmap: Image.Image) -> float:
    # discard RGB from RGBA image
    alpha_channel = np.array(segmap)[:, :, 3]
    num_pixels_nontransparent = (alpha_channel > 0).sum()
    return num_pixels_nontransparent / (segmap.width * segmap.height)


def print_progress(iteration: int, row_length: int, maximum: int):
    print(".", end="")
    if iteration % row_length == 0 or iteration == maximum:
        print(f" ({iteration} / {maximum})")

