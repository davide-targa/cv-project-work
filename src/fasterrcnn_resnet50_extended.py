import pdb
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.io import read_image
from torchvision.models import VGG16_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms import v2
from torchvision.transforms.functional import to_tensor

from utils.cv_logger import logging
from utils.datasets import PennFudanExtendedDataset

logger = logging.getLogger(f"cv.{Path(__file__).stem}")


def collate_detection(batch: List[Tuple[Tensor, Dict[str, Tensor]]]) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
    images = [img for img, _ in batch]
    targets = [t for _, t in batch]
    return images, targets


def quick_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device, score_thresh: float = 0.5) -> None:
    model.eval()
    images, _ = next(iter(loader))
    preds = model([images[0].to(device)])  # type: ignore[arg-type]
    keep = (preds[0]["scores"] > score_thresh).sum().item()
    print(f"[quick_eval] detections >= {score_thresh}: {keep}")


def get_model():
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    return model


def main() -> None:
    if torch.cuda.is_available():
        logger.info("CUDA disponibile. Uso la GPU.")
        device = torch.device("cuda")
    else:
        logger.warning("CUDA non disponibile. Uso la CPU.")
        device = torch.device("cpu")

    # --- dataset/loader ---
    dataset = PennFudanExtendedDataset()
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_detection)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_detection)

    model = get_model()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
    num_epochs = 30
    best = 999
    prevoius_file = None
    start_time = datetime.now()
    for epoch in range(1, num_epochs + 1):
        tic = time.perf_counter()
        model.train()
        for imgs, targets in train_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)  # type: ignore[arg-type]
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        toc = time.perf_counter()
        print(f"[epoch {epoch}] loss={loss.item():.3f} in {toc - tic:.1f}sec")
        quick_eval(model, val_loader, device, score_thresh=0.5)
        if loss.item() < best:
            logger.info(f"New best model found at epoch {epoch} with loss {loss.item():.3f} (previous best was {best:.3f}).")
            if prevoius_file:
                logger.info(f"Removing previous best model file {prevoius_file}.")
                Path(prevoius_file).unlink()
            outfile = f"{Path(__file__).stem}-{epoch}epoch-{loss.item():.3f}-{start_time.strftime('%Y%m%d-%H:%M')}.pth"
            torch.save(model.state_dict(), outfile)
            logger.info(f"New best model saved to {outfile} ({loss.item():.3f} < {best:.3f}).")
            prevoius_file = outfile
            best = loss.item()


if __name__ == "__main__":
    main()
