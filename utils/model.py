from segmentation_models import Unet
from segmentation_models.losses import categorical_crossentropy
from segmentation_models.metrics import iou_score, f1_score


def create_model(color_mode: str, backbone="resnet34"):
    num_channels = 1 if color_mode=="grayscale" else 3
    model = Unet(backbone, encoder_weights=None, input_shape=(512, 512, num_channels), classes=3)
    model.compile('Adam', loss=categorical_crossentropy, metrics=[iou_score, f1_score])
    return model

