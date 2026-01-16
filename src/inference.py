import argparse
import tempfile
import time
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageText
from plumbum.cmd import ffmpeg
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from torchvision.utils import draw_bounding_boxes, save_image

from utils.cv_logger import logging

logger = logging.getLogger(f"cv.{Path(__file__).stem}")

parser = argparse.ArgumentParser(
    prog="Video object detector",
    description="Object detection di persone nei video.",
)
parser.add_argument("input_video", type=str, help="Percorso al file di input.")
parser.add_argument("output_video", type=str, help="Percorso del file di output.")
args = parser.parse_args()

INPUT_VIDEO_PATH = Path(args.input_video)
OUTPUT_VIDEO_PATH = Path(args.output_video)
SCORE_THRESHOLD = 0.5

with tempfile.TemporaryDirectory() as tmpdirname:
    logger.info(f"Directory temporanea creata in {tmpdirname}")
    frames_dir = Path(tmpdirname) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    detection_frames_dir = Path(tmpdirname) / "detection_frames"
    detection_frames_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Estraggo i frame dal video {INPUT_VIDEO_PATH} in {frames_dir}...")
    ffmpeg["-i", str(INPUT_VIDEO_PATH.absolute()), f"{frames_dir / "out-%06d.png"}"]()
    logger.info("Estrazione frame completata.")

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load("/home/davide/code/cv/mine_pennfudan-extended-27epoch-0.018-20260110-00:26.pth", weights_only=True))
    model.cuda()
    model.eval()
    device = torch.device("cuda")
    frames_files = sorted(frames_dir.glob("out-*.png"))
    inference_tic = time.perf_counter()
    for idx, frame in enumerate(frames_files, start=1):
        tic = time.perf_counter()
        image = Image.open(frame).convert("RGB")
        image_tensor = to_tensor(image)
        preds = model([image_tensor.to(device)])
        boxes = preds[0]["boxes"][preds[0]["scores"] > SCORE_THRESHOLD]
        # draw = ImageDraw.Draw(image)
        # text = ImageText.Text(str(len(boxes)))
        # draw.text((10, 10), text, "#0f0")
        output_image = draw_bounding_boxes(to_tensor(image), boxes=torch.tensor(boxes, dtype=torch.float32), colors="blue", width=2)
        save_image(output_image, detection_frames_dir / f"det-{frame.name.split('-')[1]}")
        toc = time.perf_counter()
        logger.info(f"[{idx:04d}/{len(frames_files):04d}] Processato il frame {frame.name} in {toc - tic:.3f} secondi ({len(boxes)} persona/e).")
    inference_toc = time.perf_counter()
    logger.info(f"Inferenza completata in {inference_toc - inference_tic:.3f} secondi.")
    logger.info(f"Creo il video di output in {OUTPUT_VIDEO_PATH}...")
    ffmpeg["-framerate", "30", "-i", str(detection_frames_dir / "det-%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(OUTPUT_VIDEO_PATH.absolute())]()
    logger.info("Video di output creato con successo.")
    import pdb

    pdb.set_trace()
