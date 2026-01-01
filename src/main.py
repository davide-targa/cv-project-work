from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.models import VGG16_Weights
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms.functional import to_tensor

from utils.cv_logger import logging

logger = logging.getLogger(f"cv.{Path(__file__).stem}")


class PennFudanDataset(Dataset[Tuple[Tensor, Dict[str, Tensor]]]):

    URL: Path = Path("https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip")
    logger.info(f"Download del dataset Penn-Fudan da {URL}...")
    ZIP_NAME: Path = URL.name
    FOLDER_NAME: Path = ZIP_NAME.stem

    def __init__(self, root: str = "./src/data") -> None:
        super().__init__()
        self.root = Path(root)
        self.dataset_root = self.root / self.FOLDER_NAME
        self._download()
        self.images = sorted((self.dataset_root / "PNGImages").glob("*.png"))
        self.masks = sorted((self.dataset_root / "PedMasks").glob("*.png"))
        if not self.images:
            raise RuntimeError("PennFudanPed sembra vuoto: controlla la struttura delle cartelle.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        mask_np = np.array(mask)
        obj_ids = np.unique(mask_np)
        obj_ids = obj_ids[obj_ids != 0]  # 0 = sfondo

        boxes_list: List[List[float]] = []
        for obj_id in obj_ids:
            ys, xs = np.where(mask_np == obj_id)
            if xs.size == 0 or ys.size == 0:
                continue
            x_min, x_max = float(xs.min()), float(xs.max())
            y_min, y_max = float(ys.min()), float(ys.max())
            boxes_list.append([x_min, y_min, x_max, y_max])

        if boxes_list:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.ones((boxes.size(0),), dtype=torch.int64)  # unica classe: person
            wh = boxes[:, 2:] - boxes[:, :2]
            area = (wh[:, 0] * wh[:, 1]).clamp(min=1.0)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((labels.size(0),), dtype=torch.int64)

        target: Dict[str, Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }
        return to_tensor(img), target

    def _download(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.dataset_root.exists():  # Dataset già esistente
            return
        download_and_extract_archive(
            url=self.URL,
            download_root=str(self.root),
            filename=self.ZIP_NAME,
        )


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


def main() -> None:
    if torch.cuda.is_available():
        print("CUDA disponibile. Uso la GPU.")
        device = torch.device("cuda")
    else:
        print("CUDA non disponibile. Uso la CPU.")
        device = torch.device("cpu")

    # --- dataset/loader ---
    dataset = PennFudanDataset()
    generator = torch.Generator().manual_seed(0)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_detection)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_detection)

    # --- modello SSD ---
    model = ssd300_vgg16(
        weights=None,
        weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
        num_classes=2,
    )
    model.to(device)

    # --- training mini ---
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        model.train()
        for imgs, targets in train_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            """
            boxes: tensor float32 [N, 4] con le bounding box in formato (xmin, ymin, xmax, ymax) in pixel sull’immagine originale.
            labels: tensor int64 [N] con l’id di classe di ciascun box (qui sempre 1, perché Penn-Fudan ha solo “person”).
            image_id: tensor int64 [1] usato solo come identificativo interno; in questo script è l’indice dell’immagine nel dataset.
            area: tensor float32 [N] con l’area di ogni box (serve a certe metriche, SSD non lo usa direttamente).
            iscrowd: tensor int64 [N] che segnala box “crowd” (COCO-style); lo teniamo tutto a zero.
            """
            loss_dict = model(imgs, targets)  # type: ignore[arg-type]
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[epoch {epoch}] loss={loss.item():.3f}")
        quick_eval(model, val_loader, device, score_thresh=0.5)

    torch.save(model.state_dict(), "ssd_pennfudan.pth")
    print("Model saved to ssd_pennfudan.pth")


if __name__ == "__main__":
    main()
