import pdb
from pathlib import Path

from ultralytics import YOLO

from utils.cv_logger import logging
from utils.datasets import PennFudanYOLODataset

logger = logging.getLogger(f"cv.{Path(__file__).stem}")


def main():
    PennFudanYOLODataset()
    model = YOLO("yolo26n.pt")
    model.train(data="src/pf.yaml", epochs=100, imgsz=640)


if __name__ == "__main__":
    main()
