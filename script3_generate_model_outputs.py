import os
from inference import gen_outcomes_and_calc_iou
from settings import load_env
from utils.inference_utils import perform_on_all_slides

if __name__ == "__main__":
    load_env()

    level = int(os.environ['level'])
    tubule_area = int(os.environ['tubule_area'])
    color_mode = os.environ['color_mode']
    generation = int(os.environ['generation'])

    perform_on_all_slides(level, tubule_area, color_mode, generation, gen_outcomes_and_calc_iou)

    print("Script finished")
