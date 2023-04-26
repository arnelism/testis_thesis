from typing import TypedDict, Literal, List, Callable

from available_models import available_models


class InferenceConfig(TypedDict):
    slidename: str
    area_name: str
    slide_overlap: int
    level: int
    tubule_area: int
    color_mode: Literal["color", "grayscale"]
    generation: int
    model_name: str


def get_configs(level: int, tubule_area: int, color_mode: str, generation: int) -> List[InferenceConfig]:

    common = {
        "level": level,
        "tubule_area": tubule_area,
        "color_mode": color_mode,
        "generation": generation,
        "model_name": available_models[(level, tubule_area, color_mode, generation)],
    }

    return [
        {"slidename": "alpha", "area_name": "Test Region A", "slide_overlap": 25} | common,
        {"slidename": "alpha", "area_name": "Test Region A", "slide_overlap": 50} | common,

        {"slidename": "beta", "area_name": "Test Region C", "slide_overlap": 25} | common,
        {"slidename": "beta", "area_name": "Test Region C", "slide_overlap": 50} | common,

        {"slidename": "gamma", "area_name": "Test Region Gamma2", "slide_overlap": 25} | common,
        {"slidename": "gamma", "area_name": "Test Region Gamma2", "slide_overlap": 50} | common,
    ]


def perform_on_all_slides(level: int, tubule_area: int, color_mode: Literal["color", "grayscale"], generation: int, func: Callable[[InferenceConfig], None]):
    configs = get_configs(level, tubule_area, color_mode, available_models[(level, tubule_area, color_mode, generation)])
    print(f"Running callback on {len(configs)} configs\n")

    for cfg in configs:
        print(f"\nRunning processing on config: {cfg}")
        func(cfg)

    print("\nAll callbacks finished successfully")

