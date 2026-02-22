#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset COCO custom pour les images radar .tif monocanal.

Hérite de COCODataset de YOLOX en surchargeant uniquement le chargement
des images pour gérer les .tif uint8 grayscale → tensor 1 canal float32.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from pycocotools.coco import COCO

from yolox.data.datasets import COCODataset


class RadarCOCODataset(COCODataset):
    """
    Dataset COCO adapté aux images radar .tif grayscale 128x128.

    Par rapport à COCODataset standard :
    - Chargement via PIL/tifffile pour les .tif
    - Images retournées en (H, W, 1) float32 normalisées [0, 1]
    - Compatible avec le pipeline YOLOX (labels COCO bbox)

    Args:
        data_dir    : Racine du dataset, ex. "data/SOC_50classes_coco"
        ann_file    : Nom du fichier annotation, ex. "train.json"
        img_dir     : Sous-dossier des images, ex. "images/train"
        img_size    : Taille cible (H, W) pour le resize
        transform   : Transform à appliquer (RadarTrainTransform ou RadarValTransform)
        cache       : Mettre les images en mémoire RAM (déconseillé sur Colab)
    """

    def __init__(
        self,
        data_dir: str,
        ann_file: str,
        img_dir: str,
        img_size: Tuple[int, int] = (160, 160),
        transform: Optional[Callable] = None,
        cache: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.ann_path = self.data_dir / "annotations" / ann_file
        self.img_dir = self.data_dir / img_dir
        self.img_size = img_size
        self.transform = transform

        # Chargement des annotations COCO
        self.coco = COCO(str(self.ann_path))
        self.ids = sorted(self.coco.imgs.keys())
        self.class_ids = sorted(self.coco.getCatIds())

        # Mapping category_id COCO → index continu [0, num_classes-1]
        self.cat_to_label = {
            cat_id: idx for idx, cat_id in enumerate(self.class_ids)
        }

        self.annotations = self._load_coco_annotations()

        if cache:
            self._cache = self._build_cache()
        else:
            self._cache = None

    # ── Annotations ──────────────────────────────────────────────────────

    def _load_coco_annotations(self):
        """
        Charge toutes les annotations et les structure par image.
        Retourne une liste de dicts, indexée comme self.ids.
        """
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
                # Filtre les boîtes dégénérées
                if w <= 0 or h <= 0:
                    continue
                label = self.cat_to_label[ann["category_id"]]
                # Format interne : [x1, y1, x2, y2, label]
                boxes.append([x, y, x + w, y + h, label])

            annotations.append(
                {
                    "img_id": img_id,
                    "file_name": img_info["file_name"],
                    "height": img_info["height"],
                    "width": img_info["width"],
                    "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 5),
                }
            )
        return annotations

    # ── Chargement image ─────────────────────────────────────────────────

    def _load_image(self, index: int) -> np.ndarray:
        """
        Charge une image .tif en grayscale et la retourne en (H, W, 1) uint8.
        """
        ann = self.annotations[index]
        img_path = self.img_dir / ann["file_name"]

        # cv2.IMREAD_GRAYSCALE lit correctement les .tif uint8 monocanal
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Impossible de charger l'image : {img_path}")

        # Ajoute la dimension canal : (H, W) → (H, W, 1)
        img = img[:, :, np.newaxis]
        return img

    # ── Cache optionnel ──────────────────────────────────────────────────

    def _build_cache(self):
        """Charge toutes les images en RAM (à n'utiliser que si la mémoire le permet)."""
        print("Construction du cache RAM...")
        cache = []
        for i in range(len(self.ids)):
            cache.append(self._load_image(i))
        print(f"Cache construit : {len(cache)} images.")
        return cache

    # ── Interface Dataset ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        """
        Retourne (img, target, img_info, img_id).

        img    : np.ndarray (H, W, 1) uint8
        target : np.ndarray (N, 5) [x1, y1, x2, y2, label]
        """
        ann = self.annotations[index]

        if self._cache is not None:
            img = self._cache[index].copy()
        else:
            img = self._load_image(index)

        target = ann["boxes"].copy()
        img_info = (ann["height"], ann["width"])

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target, img_info, ann["img_id"]

    def pull_item(self, index: int):
        """Alias utilisé par certains évaluateurs YOLOX."""
        return self.__getitem__(index)