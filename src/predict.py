import argparse
import tempfile
import time
from importlib import import_module
from pathlib import Path

import torch
from PIL import Image
from plumbum.cmd import ffmpeg
from torchvision.transforms.functional import to_tensor
from torchvision.utils import draw_bounding_boxes, save_image

from utils.cv_logger import logging

logger = logging.getLogger(f"cv.{Path(__file__).stem}")

parser = argparse.ArgumentParser(
    prog="Video object detector",
    description="Object detection di persone nei video.",
)
parser.add_argument(
    "model_name",
    type=str,
    help="Il modulo python da cui importare il modello",
    choices=[model.stem for model in Path("src/models").glob("*.py") if model.stem != "__init__"],
)
parser.add_argument(
    "weights_file",
    type=str,
    help="Il file contenente i pesi del modello",
)
parser.add_argument(
    "input_video",
    type=str,
    help="Percorso al file di input.",
)
parser.add_argument(
    "output_video",
    type=str,
    help="Percorso del file di output.",
)
args = parser.parse_args()

INPUT_VIDEO_PATH = Path(args.input_video)
OUTPUT_VIDEO_PATH = Path(args.output_video)
SCORE_THRESHOLD = 0.5

# Import dinamico del model in base al parametro da riga di comando
get_model = getattr(import_module(f"models.{args.model_name}"), "get_model")
with tempfile.TemporaryDirectory() as tmpdirname:
    logger.info(f"Directory temporanea creata in {tmpdirname}")
    frames_dir = Path(tmpdirname) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    detection_frames_dir = Path(tmpdirname) / "detection_frames"
    detection_frames_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Estraggo i frame dal video {INPUT_VIDEO_PATH} in {frames_dir}...")
    ffmpeg["-i", str(INPUT_VIDEO_PATH.absolute()), f"{frames_dir / "out-%06d.png"}"]()
    logger.info("Estrazione frame completata.")

    model = get_model()
    model.load_state_dict(torch.load(Path(args.weights_file), weights_only=True))
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
        output_image = draw_bounding_boxes(to_tensor(image), boxes=torch.tensor(boxes, dtype=torch.float32), colors="blue", width=5)
        save_image(output_image, detection_frames_dir / f"det-{frame.name.split('-')[1]}")
        toc = time.perf_counter()
        logger.info(f"[{idx:04d}/{len(frames_files):04d}] Processato il frame {frame.name} in {toc - tic:.3f} secondi ({len(boxes)} persona/e).")
    inference_toc = time.perf_counter()
    logger.info(f"Inferenza completata in {inference_toc - inference_tic:.3f} secondi.")
    logger.info(f"Creo il video di output in {OUTPUT_VIDEO_PATH}...")
    ffmpeg["-framerate", "30", "-i", str(detection_frames_dir / "det-%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(OUTPUT_VIDEO_PATH.absolute())]()
    logger.info("Video di output creato con successo.")
