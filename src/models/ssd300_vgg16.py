from pathlib import Path

from torchvision.models.detection import ssd300_vgg16

from utils.cv_logger import logging

logger = logging.getLogger(f"cv.{Path(__file__).stem}")


def get_model():
    model = ssd300_vgg16(weights="DEFAULT")
    return model
