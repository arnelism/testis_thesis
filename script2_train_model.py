# Script for training a model or many models (step 2 of 3)

# Loads image slices and fits a segmentation model on them
# Saves model parameters and logs

# Runs for a long time. Should be called via slurm_train.sh

import dataclasses
from typing import Optional

from segmentation_models import Unet
from segmentation_models.losses import categorical_crossentropy
from segmentation_models.metrics import iou_score, f1_score
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from datetime import datetime
import wandb
from wandb.keras import WandbCallback

from settings import load_env

from utils.image_loader import get_image_loader


@dataclasses.dataclass
class Args:
    level: int
    overlap: int
    color_mode: str
    backbone: str
    epochs: int
    batch_size: int

    train_slidefile: str
    test_slidefile: str

    enable_wb: bool
    train_size: int
    test_size: int

    name: Optional[str]


# noinspection PyTypeChecker
def load_image(filename, color_mode):
    img = Image.open(filename)
    if color_mode == "grayscale":
        img = img.convert('L')
        img = np.array(img)
    else:
        img = np.array(img)
        img = img[..., 0:-1]  # remove alpha layer

    return img


def load_mask(filename):
    img = Image.open(filename)
    # noinspection PyTypeChecker
    mask = np.array(img)
    mask = mask[..., [0, 1, 3]]
    mask[..., 0] = (mask[..., 0]/255)
    mask[..., 1] = (mask[..., 1]/128)
    mask[..., 2] = ((255-mask[..., 2])/255)
    mask = mask.astype(float)
    return mask


def load_dataset(folder, color_mode, max_size=-1):
    images = []
    masks = []
    print(f"Loading dataset {folder}")
    i = 0
    for img in os.listdir(f"{folder}"):
        if max_size is not None and max_size <= i:
            break

        if img.endswith("_orig.png"):
            images.append(
                load_image(f"{folder}/{img}", color_mode)
            )
            seg = img[0:-9] + "_seg.png"
            masks.append(
                load_mask(f"{folder}/{seg}")
            )

            i = i+1
            print(".", end="")
            if i % 100 == 0:
                print(f"\t{i}")

    print("")
    images = np.array(images)
    masks = np.array(masks)
    return images, masks


def create_model(color_mode, backbone="resnet34"):
    print(f"Creating model (color_mode={color_mode}, backbone={backbone})")
    try:
        num_channels = 1 if color_mode == "grayscale" else 3
        model = Unet(backbone, encoder_weights=None, input_shape=(512, 512, num_channels), classes=3)
        model.compile('Adam', loss=categorical_crossentropy, metrics=[iou_score, f1_score])
        return model
    except Exception as e:
        print("Error creating model")
        raise e


def get_dataset_path(cfg: Args, mode: str) -> str:
    # if cfg.train_slidefile != cfg.test_slidefile:
    #     mid = "trainonly/" if mode == "train" else "testonly/"
    # else:
    #     mid = ""

    slidefile = cfg.train_slidefile if mode == "train" else cfg.test_slidefile
    if slidefile == "all":
        slidefile = "*"

    return f"{os.environ['slides']}/{slidefile}/level{cfg.level}_overlap{cfg.overlap}/{mode}"


def get_date_str(now=datetime.now()):
    return now.isoformat("_", "minutes").replace(":","")


def run_pipeline(cfg: Args):
    level = cfg.level
    overlap = cfg.overlap
    color_mode = cfg.color_mode
    backbone = cfg.backbone
    epochs = cfg.epochs
    batch_size = cfg.batch_size


    # create data loader
    train_data = get_image_loader(get_dataset_path(cfg, "train"), cfg.batch_size, color_mode=color_mode)
    test_data = get_image_loader(get_dataset_path(cfg, "test"), cfg.batch_size, color_mode=color_mode)
    print(f"Loaded dataset (level={level}, overlap={overlap})")

    # create model
    model_name = f"{cfg.train_slidefile}{cfg.test_slidefile}_level{level}_overlap{overlap}_{color_mode}_{get_date_str()}"
    if args.name is not None:
        model_name = f"{args.name}_{model_name}"

    model_path = f"{os.environ['models']}/{model_name}"
    model = create_model(color_mode, backbone)

    # configure callbacks
    callbacks = []

    # ... save weights for best model
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{model_path}/checkpoint.ckpt",
            save_weights_only=True,
            save_freq='epoch',
            save_best_only=True
        )
    )

    # ... save tensorboard logs
    log_dir = f"{os.environ['train_logs']}/{model_name}/{get_date_str()}"
    print(f"Saving logs to {log_dir}")
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir
        )
    )

    # ... save wb logs
    if cfg.enable_wb:
        config = {
            "generation": 2,
            "level": level,
            "overlap": overlap,
            "color_mode": color_mode,
            "backbone": backbone,
            "batch_size": batch_size,
            "epochs": epochs
        }
        os.environ["WANDB_NOTEBOOK_NAME"] = "train_model.py"
        wandb.init(project="Testis Thesis", entity="arnelism", config=config)
        callbacks.append(
            WandbCallback()
        )

    print(model.summary())

    # train the model
    if epochs > 0:
        model.fit(
            train_data,
            validation_data=test_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

    # wrap-up
    model.save_weights(f"{model_path}/final.ckpt")

    if cfg.enable_wb:
        wandb.finish()

    print("\nFINISHED TRAINING MODEL\n")


########################################


if __name__ == "__main__":
    load_env()

    args = Args(
        level=int(os.environ['level']),
        overlap=int(os.environ['overlap']),
        color_mode=os.environ['color_mode'],
        backbone=os.environ['backbone'],
        epochs=int(os.environ['train_epochs']),
        batch_size=int(os.environ['batch_size']),

        train_slidefile=os.environ['train_slidefile'],
        test_slidefile=os.environ['test_slidefile'],
        enable_wb=bool(os.environ.get('enable_wb') == 'true'),

        train_size=int(os.environ.get('train_size')) if os.environ.get('train_size') is not None else None,
        test_size=int(os.environ.get('test_size')) if os.environ.get('test_size') is not None else None,

        name=os.environ.get('name'),
    )

    print(f"\n\nRunning pipeline: {args}")

    if tf.test.is_gpu_available():
        print("GPU is available, ready to go")
    else:
        raise Exception("GPU not found. Cannot continue")

    run_pipeline(args)

    print("\n\nPIPELINE COMPLETE!\n")
