# Script for training a model or many models (step 2 of 3)

# Loads image slices and fits a segmentation model on them
# Saves model parameters and logs

# Runs for a long time. Should be called via slurm_train.sh

import argparse

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
        if max_size != -1 and max_size <= i:
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

    images = np.array(images)
    masks = np.array(masks)
    return images, masks


def create_model(color_mode, backbone="resnet34"):
    num_channels = 1 if color_mode == "grayscale" else 3
    model = Unet(backbone, encoder_weights=None, input_shape=(512, 512, num_channels), classes=3)
    model.compile('Adam', loss=categorical_crossentropy, metrics=[iou_score, f1_score])
    return model


def get_dataset_path(cfg: Args, mode: str) -> str:
    if cfg.train_slidefile != cfg.test_slidefile:
        mid = "trainonly/" if mode == "train" else "testonly/"
    else:
        mid = ""

    slidefile = cfg.train_slidefile if mode == "train" else cfg.test_slidefile
    full_slidefile = os.environ[f"slidefile_{slidefile}"]

    return f"{os.environ['slides']}/{mid}{full_slidefile}/level{args.level}_overlap{args.overlap}/{mode}"


def run_pipeline(cfg: Args):
    level = cfg.level
    overlap = cfg.overlap
    color_mode = cfg.color_mode
    backbone = cfg.backbone
    epochs = cfg.epochs
    batch_size = cfg.batch_size


    # TODO convert to tf image loader!
    train_images, train_masks = load_dataset(
        get_dataset_path(cfg, "train"),
        color_mode,
        cfg.train_size
    )
    test_images, test_masks = load_dataset(
        get_dataset_path(cfg, "test"),
        color_mode,
        cfg.test_size
    )
    print(f"Loaded dataset (level={level}, overlap={overlap})")
    print((train_images.shape, train_masks.shape, test_images.shape, test_masks.shape))

    model_name = f"{cfg.train_slidefile}{cfg.test_slidefile}_level{level}_overlap{overlap}_col{color_mode}_{backbone}_epochs{epochs}"
    model_path = f"{os.environ['models']}/{model_name}"
    model = create_model(color_mode, backbone)

    callbacks = []

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{model_path}/checkpoint.ckpt",
            save_weights_only=True,
            save_freq='epoch',
            save_best_only=True
        )
    )

    now = datetime.now().strftime("%Y-%m-%d_%H%M")
    logs_folder = os.environ["train_logs"]
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=f"{logs_folder}/{model_name}/{now}"
        )
    )

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

    model.fit(
        x=train_images,
        y=train_masks,
        validation_data=(test_images, test_masks),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    model.save_weights(f"{model_path}/final.ckpt")

    if cfg.enable_wb:
        wandb.finish()

    print("\nFINISHED TRAINING MODEL\n")
    return model, train_images, train_masks, test_images, test_masks


########################################


if __name__ == "__main__":
    load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--color_mode")
    parser.add_argument("--backbone", default="resnet34")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=64, type=int)

    parser.add_argument("--train_slidefile", type=str)
    parser.add_argument("--test_slidefile", type=str)

    parser.add_argument('--enable_wb', action="store_true")

    parser.add_argument('--train_size', type=int, default=-1)
    parser.add_argument('--test_size', type=int, default=-1)

    # noinspection PyTypeChecker
    args: Args = parser.parse_args()


    print(f"\n\nRunning pipeline: {args}")

    if tf.test.is_gpu_available():
        print("GPU is available, ready to go")
    else:
        raise Exception("GPU not found. Cannot continue")

    run_pipeline(args)

    print("\n\nPIPELINE COMPLETE!\n")
