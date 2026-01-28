import argparse
from pathlib import Path

from ultralytics import YOLO

parser = argparse.ArgumentParser(
    prog="Video object detection with YOLO",
    description="Object detection di persone nei video con YOLO.",
)
parser.add_argument("input_video", type=str, help="Percorso al file di input.")
args = parser.parse_args()

detections_path = Path(__file__).parent.parent / "runs" / "detect" / "train" / "weights"
model = YOLO(detections_path / "best.pt")
results = model(args.input_video, save=True, show_labels=False)
# results = model(args.input_video, save=True, show=True, show_labels=False)
