import os
from typing import Literal

import skimage
from PIL import Image
from sklearn.metrics import jaccard_score

from available_models import available_models
from inference import InferenceConfig
from post_processing import post_process
from settings import load_env
from utils.inference_utils import perform_on_all_slides


if __name__ == "__main__":
    load_env()

    level = int(os.environ['level'])
    tubule_area = int(os.environ['tubule_area'])
    color_mode: Literal["color", "grayscale"] = os.environ['color_mode']
    generation = int(os.environ['generation'])

    model_name = available_models[(level, tubule_area, color_mode, generation)]


    def postproc_and_iou(cfg: InferenceConfig):
        print(f"Performing post-processing: {cfg}")

        pred_filename_base = f"output/{model_name}/composite.{cfg['slidename']}.{cfg['slide_overlap']}"
        y_pred = skimage.io.imread(f"{pred_filename_base}.clean.png")
        y_true = skimage.io.imread(f"work/inference/{cfg['slidename']}/ground_truth_level{cfg['level']}_model.png")

        y_processed = post_process(y_pred, thresholding="otsu", level=cfg['level'])

        # save post-processed image and calculate iou
        print("Saving image")
        img = Image.fromarray(y_processed)
        img.save(f"{pred_filename_base}.postproc.png")

        print("calculating iou")
        if y_true.shape != y_processed.shape:
            raise Exception(f"Shape mismatch! {y_true.shape} vs {y_processed.shape}")

        iou = round(jaccard_score(y_true.ravel(), y_processed.ravel(), average='weighted'), 3)
        print(f"IoU:{iou}")

        target = f"output/{cfg['model_name']}/iou.postproc.{cfg['slidename']}.{cfg['slide_overlap']}.txt"
        with open(target, "w") as f:
            f.write(str(iou))


    perform_on_all_slides(level, tubule_area, color_mode, generation, postproc_and_iou)

    print("Script finished")
