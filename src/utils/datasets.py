from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, random_split
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms.functional import to_tensor

from utils.cv_logger import logging

logger = logging.getLogger(f"cv.{Path(__file__).stem}")


class PennFudanDataset(Dataset):

    URL: str = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    ZIP_NAME: str = URL.split("/")[-1]
    FOLDER_NAME: str = ZIP_NAME.split(".")[0]

    def __init__(self, root: str = "./src/data") -> None:
        super().__init__()
        logger.info("Utilizzo il dataset base")
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
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        mask_np = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # 0 = sfondo

        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(image.permute(1, 2, 0))
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask)
        # plt.show()

        # Estrazione delle maschere di ogni persona all'interno dell'immagine
        boxes_list = []
        for obj_id in obj_ids:
            ys, xs = np.where(mask_np == obj_id)
            if xs.size == 0 or ys.size == 0:
                continue
            x_min, x_max = float(xs.min()), float(xs.max())
            y_min, y_max = float(ys.min()), float(ys.max())
            boxes_list.append([x_min, y_min, x_max, y_max])

        # from torchvision.utils import draw_bounding_boxes
        # output_image = draw_bounding_boxes(to_tensor(image), boxes=torch.tensor(boxes_list, dtype=torch.float32), colors="red", width=2)
        # plt.figure(figsize=(12, 12))
        # plt.imshow(output_image.permute(1, 2, 0))

        if boxes_list:
            # Coordinates delle bounding box per ogni maschera dell'immagine
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            # La classe è unica quindi per  ad ogni persona assegniamo label 1
            labels = torch.ones((boxes.size(0),), dtype=torch.int64)  # unica classe: person
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return to_tensor(image), target

    def _download(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.dataset_root.exists():  # Dataset già esistente
            logger.info("Dataset Penn-Fudan già presente.")
            return
        logger.info("Download ed estrazione del dataset Penn-Fudan...")
        download_and_extract_archive(url=self.URL, download_root=str(self.root), filename=self.ZIP_NAME)


class PennFudanTextDataset(PennFudanDataset):
    def __init__(self, root: str = "./src/data") -> None:
        super().__init__(root)
        logger.info("Utilizzo il dataset text")
        self.annotations_root = self.dataset_root / "Annotation"
        self.boxes = self._get_image_data()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        image = Image.open(self.images[idx]).convert("RGB")
        boxes_list = self.boxes.loc[self.boxes["filename"] == self.images[idx].name, ["x_min", "y_min", "x_max", "y_max"]].values.tolist()
        if boxes_list:
            # Coordinate delle bounding box per ogni maschera dell'immagine
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            # La classe è unica quindi a ogni persona assegniamo label 1
            labels = torch.ones((boxes.size(0),), dtype=torch.int64)  # unica classe: person
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return to_tensor(image), target

    def _get_image_data(self):
        image_data = []
        for idx, file_path in enumerate(sorted(self.annotations_root.glob("*.txt")), start=1):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.readlines()
                for line in content:
                    if line.startswith("Image filename :"):
                        filename = Path(line.split(":")[1].strip().strip('"')).name

                    if line.startswith("Image size"):
                        width, height, _ = line.split(":")[1].strip().split(" x ")

                    if line.startswith("Bounding box for object"):
                        object_index = int(line.split(" ")[4])
                        coords = line.split(": ")[1].strip().split(" - ")
                        x_min = int(coords[0].split(", ")[0].lstrip("("))
                        y_min = int(coords[0].split(", ")[1].rstrip(")"))
                        x_max = int(coords[1].split(", ")[0].lstrip("("))
                        y_max = int(coords[1].split(", ")[1].rstrip(")"))
                        image_data.append(
                            {
                                "filename": filename,
                                "object_index": object_index,
                                "x_min": x_min,
                                "y_min": y_min,
                                "x_max": x_max,
                                "y_max": y_max,
                                "width": int(width),
                                "height": int(height),
                                "yolo_x_center": (x_min + x_max) / 2 / int(width),
                                "yolo_y_center": (y_min + y_max) / 2 / int(height),
                                "yolo_width": (x_max - x_min) / int(width),
                                "yolo_height": (y_max - y_min) / int(height),
                            }
                        )
                        logger.debug(f"{idx} - {filename:<16} ({object_index}): ({x_min}, {y_min}) - ({x_max}, {y_max})")
        logger.info(f"Trovati {len(image_data)} oggetti in {len(self.images)} immagini.")
        return pd.DataFrame(image_data)


class PennFudanYOLODataset(PennFudanTextDataset):

    PF_IMAGES_PATH = Path("src/data/PennFudanPed/PNGImages/")
    PF_ANNOTATIONS_PATH = Path("src/data/PennFudanPed/Annotation/")
    YOLO_IMAGES_PATH = Path("src/data/PennFudan_4_YOLO/images/")
    YOLO_LABELS_PATH = Path("src/data/PennFudan_4_YOLO/labels/")
    YOLO_IMAGES_TRAIN_PATH = YOLO_IMAGES_PATH / "train"
    YOLO_IMAGES_VAL_PATH = YOLO_IMAGES_PATH / "val"
    YOLO_LABELS_TRAIN_PATH = YOLO_LABELS_PATH / "train"
    YOLO_LABELS_VAL_PATH = YOLO_LABELS_PATH / "val"
    TRAIN_RATIO = 0.8

    def __init__(self):
        super().__init__()
        self.train_size = int(0.8 * len(self.images))
        self.val_size = int(len(self.images) - self.train_size)
        self._prepare_yolo_dataset()

    def _prepare_yolo_dataset(self):
        logger.info("Creazione delle directory...")
        rmtree(self.YOLO_IMAGES_PATH)
        rmtree(self.YOLO_LABELS_PATH)
        Path.mkdir(self.YOLO_IMAGES_TRAIN_PATH, parents=True, exist_ok=True)
        Path.mkdir(self.YOLO_IMAGES_VAL_PATH, parents=True, exist_ok=True)
        Path.mkdir(self.YOLO_LABELS_TRAIN_PATH, parents=True, exist_ok=True)
        Path.mkdir(self.YOLO_LABELS_VAL_PATH, parents=True, exist_ok=True)

        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(self.images, [self.train_size, self.val_size], generator=generator)

        logger.info("Preparazione del dataset di training per YOLO...")
        for image_path in train_ds:
            # Copia l'immagine nella cartella di training
            dest_image_path = self.YOLO_IMAGES_TRAIN_PATH / image_path.name
            dest_image_path.write_bytes(image_path.read_bytes())
            logger.debug(f"Copiata l'immagine: {image_path}")

            # Crezione del file delle label in formato YOLO
            boxes_df = self.boxes.loc[self.boxes["filename"] == image_path.name]
            yolo_labels = []
            for _, row in boxes_df.iterrows():
                yolo_labels.append(f"0 {row['yolo_x_center']} {row['yolo_y_center']} {row['yolo_width']} {row['yolo_height']}\n")
            label_file_path = self.YOLO_LABELS_TRAIN_PATH / (image_path.stem + ".txt")
            with open(label_file_path, "w", encoding="utf-8") as label_file:
                label_file.writelines(yolo_labels)
            logger.debug(f"Creato il file di label {label_file_path}")

        logger.info("Preparazione del dataset di validazione per YOLO...")
        for image_path in val_ds:
            # Copia l'immagine nella cartella di validazione
            dest_image_path = self.YOLO_IMAGES_VAL_PATH / image_path.name
            dest_image_path.write_bytes(image_path.read_bytes())
            logger.debug(f"Copiata l'immagine: {image_path}")

            # Creazione del file delle label in formato YOLO
            boxes_df = self.boxes.loc[self.boxes["filename"] == image_path.name]
            yolo_labels = []
            for _, row in boxes_df.iterrows():
                yolo_labels.append(f"0 {row['yolo_x_center']} {row['yolo_y_center']} {row['yolo_width']} {row['yolo_height']}\n")
            label_file_path = self.YOLO_LABELS_VAL_PATH / (image_path.stem + ".txt")
            with open(label_file_path, "w", encoding="utf-8") as label_file:
                label_file.writelines(yolo_labels)
            logger.debug(f"Creato il file di label {label_file_path}")


def rmtree(root):
    for p in root.iterdir():
        if p.is_dir():
            rmtree(p)
        else:
            p.unlink()

    root.rmdir()
