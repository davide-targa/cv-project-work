from pathlib import Path

from ultralytics import YOLO

detections_path = Path(__file__).parent.parent / "runs" / "detect" / "train" / "weights"
model = YOLO(detections_path / "best.pt")
# results = model(["src/testimage.jpg"])
results = model("/home/davide/code/cv/src/inference/pedestrian/pedestrians_video.mp4", save=True, show=True)
