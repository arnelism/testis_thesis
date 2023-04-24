import os
from available_models import available_models
from inference import run_model_on_all_slides
from settings import load_env


if __name__ == "__main__":
    load_env()

    level = int(os.environ['level'])
    tubule_area = int(os.environ['tubule_area'])
    color_mode = os.environ['color_mode']
    model_name = available_models[(level, tubule_area, color_mode)]

    run_model_on_all_slides(level, tubule_area, color_mode)


    print("Script finished")
