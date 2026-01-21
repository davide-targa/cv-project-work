import argparse
import time
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split

from utils.cv_logger import logging
from utils.datasets import PennFudanDataset, PennFudanExtendedDataset

logger = logging.getLogger(f"cv.{Path(__file__).stem}")


def quick_eval(model: torch.nn.Module, loader: DataLoader, device: torch.device, score_thresh: float = 0.5) -> None:
    model.eval()
    images, _ = next(iter(loader))
    preds = model([images[0].to(device)])  # type: ignore[arg-type]
    keep = (preds[0]["scores"] > score_thresh).sum().item()
    print(f"[quick_eval] detections >= {score_thresh}: {keep}")


def collate_detection(batch: List[Tuple[Tensor, Dict[str, Tensor]]]) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
    images = [img for img, _ in batch]
    targets = [t for _, t in batch]
    return images, targets


def main(num_epochs: int, model_module: str, dataset: torch.utils.data.Dataset) -> None:
    if torch.cuda.is_available():
        logger.info("CUDA disponibile. Uso la GPU.")
        device = torch.device("cuda")
    else:
        logger.warning("CUDA non disponibile. Uso la CPU.")
        device = torch.device("cpu")

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_detection)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_detection)

    get_model = getattr(import_module(f"models.{model_module}"), "get_model")
    model = get_model()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
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
    parser = argparse.ArgumentParser(
        prog="Addestramento rete per object detection",
        description="Addestramento di una rete per object detection di persone nei video.",
    )
    parser.add_argument(
        "num_epochs",
        type=int,
        help="Il numero di epoche di addestramento",
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Il modulo del modello da addestrare",
        choices=[model.stem for model in Path("src/models").glob("*.py") if model.stem != "__init__"],
    )
    parser.add_argument(
        "dataset_type",
        type=str,
        help="Il tipo di dataset da utilizzare",
        choices=["base", "extended"],
    )
    args = parser.parse_args()
    match args.dataset_type:
        case "base":
            dataset = PennFudanDataset()
        case "extended":
            dataset = PennFudanExtendedDataset()
    main(num_epochs=args.num_epochs, model_module=args.model_name, dataset=dataset)
