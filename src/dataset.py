import os.path
import random
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from pycocotools.coco import COCO
from yolox.data.datasets import COCODataset


class SAR_ATR_Dataset(COCODataset):
    def __init__(
        self,
        root: Union[str, Path],
        annFile: str,
        transforms=None,
        subset_ratio: float = 1.0,
        img_size: tuple = (160, 160),
    ) -> None:
        # COCODataset.__init__ attend : data_dir, json_file, name, img_size
        # On l'initialise avec des valeurs minimales puis on écrase
        self.img_size = img_size  # requis par COCODataset avant super().__init__
        super().__init__(
            data_dir=str(root),
            json_file=annFile,
            name="",           # on gère les chemins nous-mêmes
            img_size=img_size,
            preproc=None,
        )
        # Surcharger avec notre propre chargement COCO
        self.root = str(root)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self._transforms = transforms

        self.class_ids = sorted(self.coco.getCatIds())
        self.cat_to_label = {
            cat_id: idx for idx, cat_id in enumerate(self.class_ids)
        }

        if subset_ratio < 1.0:
            n_samples = int(len(self.ids) * subset_ratio)
            random.seed(94)
            self.ids = random.sample(self.ids, n_samples)

        self.annotations = self._load_all_annotations()

    def _load_all_annotations(self):
        annotations = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            img_info = self.coco.imgs[img_id]
            boxes = []
            for ann in anns:
                if ann.get("ignore", 0):
                    continue
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue
                label = self.cat_to_label[ann["category_id"]]
                boxes.append([x, y, x + w, y + h, label])
            annotations.append({
                "img_id": img_id,
                "file_name": img_info["file_name"],
                "height": img_info["height"],
                "width": img_info["width"],
                "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 5),
            })
        return annotations

    def _load_image(self, index: int) -> np.ndarray:
        file_name = self.annotations[index]["file_name"]
        path = os.path.join(self.root, file_name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image introuvable : {path}")
        return np.stack([img, img, img], axis=-1)  # (H, W, 3)

    def __getitem__(self, index: int):
        ann = self.annotations[index]
        img = self._load_image(index)
        target = ann["boxes"].copy()
        img_info = (ann["height"], ann["width"])
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target, img_info, ann["img_id"]

    def __len__(self) -> int:
        return len(self.ids)

    def pull_item(self, index: int):
        return self.__getitem__(index)