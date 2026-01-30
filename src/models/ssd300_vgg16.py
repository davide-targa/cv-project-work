from pathlib import Path

from torchvision.models.detection import _utils, ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

from utils.cv_logger import logging

logger = logging.getLogger(f"cv.{Path(__file__).stem}")


def get_model():
    model = ssd300_vgg16(weights="DEFAULT")
    in_channels = _utils.retrieve_out_channels(model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=2,
    )
    import pdb

    pdb.set_trace()
    return model
