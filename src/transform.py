#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augmentations adaptées aux images radar grayscale 128x128.

Stratégie :
- On évite les augmentations couleur (HSV, jitter) inutiles en grayscale
- On garde des transformations géométriques légères cohérentes avec
  la physique radar (symétries, rotations modérées)
- Les bounding boxes sont transformées conjointement aux images
- Format entrée/sortie images : (H, W, 1) uint8 np.ndarray
- Format labels : (N, 5) float32 [x1, y1, x2, y2, class_label]
"""

from typing import Optional, Tuple

import cv2
import numpy as np


# ── Utilitaires internes ──────────────────────────────────────────────────────

def _clip_boxes(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    """Restreint les boîtes dans les limites de l'image."""
    if len(boxes) == 0:
        return boxes
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
    return boxes


def _filter_boxes(boxes: np.ndarray, min_area: float = 4.0) -> np.ndarray:
    """Supprime les boîtes dont l'aire est trop petite après transformation."""
    if len(boxes) == 0:
        return boxes
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w * h) >= min_area
    return boxes[keep]


def _resize_image_and_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    target_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize image (H, W, 1) vers target_size (th, tw).
    Adapte les boîtes proportionnellement.
    """
    src_h, src_w = img.shape[:2]
    th, tw = target_size

    img_resized = cv2.resize(img.squeeze(-1), (tw, th), interpolation=cv2.INTER_LINEAR)
    img_resized = img_resized[:, :, np.newaxis]

    if len(boxes) > 0:
        scale_x = tw / src_w
        scale_y = th / src_h
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

    return img_resized, boxes


# ── Augmentations individuelles ───────────────────────────────────────────────

def random_flip_horizontal(
    img: np.ndarray,
    boxes: np.ndarray,
    prob: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flip horizontal aléatoire."""
    if np.random.rand() > prob:
        return img, boxes

    img = img[:, ::-1, :].copy()
    if len(boxes) > 0:
        w = img.shape[1]
        boxes = boxes.copy()
        x1 = w - boxes[:, 2]
        x2 = w - boxes[:, 0]
        boxes[:, 0] = x1
        boxes[:, 2] = x2

    return img, boxes


def random_flip_vertical(
    img: np.ndarray,
    boxes: np.ndarray,
    prob: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flip vertical aléatoire (cohérent physiquement en radar SAR)."""
    if np.random.rand() > prob:
        return img, boxes

    img = img[::-1, :, :].copy()
    if len(boxes) > 0:
        h = img.shape[0]
        boxes = boxes.copy()
        y1 = h - boxes[:, 3]
        y2 = h - boxes[:, 1]
        boxes[:, 1] = y1
        boxes[:, 3] = y2

    return img, boxes


def random_affine(
    img: np.ndarray,
    boxes: np.ndarray,
    degrees: float = 10.0,
    translate: float = 0.1,
    scale: Tuple[float, float] = (0.8, 1.2),
    shear: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transformation affine aléatoire : rotation, translation, zoom, cisaillement.
    Les boîtes sont transformées via les 4 coins puis re-englobées (axis-aligned).
    """
    h, w = img.shape[:2]
    center = np.array([w / 2, h / 2])

    # Matrice de rotation + scale
    angle = np.random.uniform(-degrees, degrees)
    s = np.random.uniform(scale[0], scale[1])
    M_rot = cv2.getRotationMatrix2D(tuple(center), angle, s)

    # Translation
    tx = np.random.uniform(-translate, translate) * w
    ty = np.random.uniform(-translate, translate) * h
    M_rot[0, 2] += tx
    M_rot[1, 2] += ty

    # Cisaillement (shear en x)
    shear_x = np.random.uniform(-shear, shear)
    M_shear = np.array([[1, np.tan(np.radians(shear_x)), 0],
                         [0, 1, 0]], dtype=np.float32)

    # Application à l'image
    img_out = cv2.warpAffine(
        img.squeeze(-1),
        M_rot,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    img_out = cv2.warpAffine(
        img_out,
        M_shear,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    img_out = img_out[:, :, np.newaxis]

    # Application aux boîtes via transformation des 4 coins
    if len(boxes) > 0:
        n = len(boxes)
        # Construire les 4 coins de chaque boîte
        corners = np.ones((n * 4, 3), dtype=np.float32)
        corners[0::4, :2] = boxes[:, [0, 1]]  # top-left
        corners[1::4, :2] = boxes[:, [2, 1]]  # top-right
        corners[2::4, :2] = boxes[:, [2, 3]]  # bottom-right
        corners[3::4, :2] = boxes[:, [0, 3]]  # bottom-left

        # Appliquer rotation
        corners_t = (M_rot @ corners.T).T
        # Appliquer shear
        corners_t = (M_shear @ np.hstack([corners_t, np.ones((n * 4, 1))]).T).T

        # Re-englobage axis-aligned
        corners_t = corners_t.reshape(n, 4, 2)
        x_min = corners_t[:, :, 0].min(axis=1)
        y_min = corners_t[:, :, 1].min(axis=1)
        x_max = corners_t[:, :, 0].max(axis=1)
        y_max = corners_t[:, :, 1].max(axis=1)

        boxes = boxes.copy()
        boxes[:, 0] = x_min
        boxes[:, 1] = y_min
        boxes[:, 2] = x_max
        boxes[:, 3] = y_max

        boxes = _clip_boxes(boxes, h, w)
        boxes = _filter_boxes(boxes)

    return img_out, boxes


def random_brightness(
    img: np.ndarray,
    delta: float = 30.0,
) -> np.ndarray:
    """Ajustement aléatoire de la luminosité (pertinent en radar pour le bruit)."""
    shift = np.random.uniform(-delta, delta)
    img = img.astype(np.float32) + shift
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def add_gaussian_noise(
    img: np.ndarray,
    sigma_range: Tuple[float, float] = (0.0, 10.0),
) -> np.ndarray:
    """
    Ajoute du bruit gaussien à l'image radar.
    Simule les variations du bruit de speckle radar.
    """
    sigma = np.random.uniform(*sigma_range)
    if sigma == 0:
        return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img_noisy = img.astype(np.float32) + noise
    return np.clip(img_noisy, 0, 255).astype(np.uint8)


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalise l'image uint8 (H, W, 1) → float32 (H, W, 1) dans [0, 1].
    Puis duplique sur 3 canaux car le backbone YOLOX attend (H, W, 3).
    → Retourne (H, W, 3) float32, compatible torchvision/YOLOX.
    """
    img_f = img.astype(np.float32) / 255.0
    # Répétition sur 3 canaux pour compatibilité backbone
    img_3c = np.repeat(img_f, 3, axis=2)
    return img_3c


# ── Transforms composites ─────────────────────────────────────────────────────

class RadarTrainTransform:
    """
    Pipeline d'augmentation pour l'entraînement.

    Ordre :
    1. Resize vers la taille cible
    2. Flip horizontal (prob configurable)
    3. Flip vertical (prob fixe 0.3)
    4. Transformation affine légère
    5. Ajustement de luminosité
    6. Bruit gaussien léger
    7. Normalisation → (H, W, 3) float32
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (160, 160),
        flip_prob: float = 0.5,
        degrees: float = 10.0,
        translate: float = 0.1,
        scale: Tuple[float, float] = (0.8, 1.2),
        shear: float = 2.0,
    ):
        self.img_size = img_size
        self.flip_prob = flip_prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Resize
        img, boxes = _resize_image_and_boxes(img, boxes, self.img_size)

        # 2. Flip horizontal
        img, boxes = random_flip_horizontal(img, boxes, self.flip_prob)

        # 3. Flip vertical
        img, boxes = random_flip_vertical(img, boxes, prob=0.3)

        # 4. Affine
        img, boxes = random_affine(
            img, boxes,
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
        )

        # 5. Luminosité
        img = random_brightness(img, delta=20.0)

        # 6. Bruit gaussien
        img = add_gaussian_noise(img, sigma_range=(0.0, 8.0))

        # 7. Normalisation
        img = normalize(img)

        return img, boxes


class RadarValTransform:
    """
    Pipeline de validation : uniquement resize + normalisation.
    Aucune augmentation aléatoire.
    """

    def __init__(self, img_size: Tuple[int, int] = (160, 160)):
        self.img_size = img_size

    def __call__(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        img, boxes = _resize_image_and_boxes(img, boxes, self.img_size)
        img = normalize(img)
        return img, boxes