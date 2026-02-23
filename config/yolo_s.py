#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fichier de configuration YOLOX pour la détection de cibles radar.
Basé sur le système d'Exp de la librairie officielle Megvii YOLOX.

Images : 128x128, grayscale (1 canal), uint8
Classes : 50 classes de cibles radar
Modèle  : YOLOX-S
"""

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # ── Identification ────────────────────────────────────────────────
        self.exp_name = "yolox_s_radar_50cls"

        # ── Modèle ────────────────────────────────────────────────────────
        # YOLOX-S : depth=0.33, width=0.50
        self.depth = 0.33
        self.width = 0.50
        self.act = "silu"

        # Images radar monocanal → on adapte le nombre de canaux en entrée
        # La modification effective se fait dans get_model() ci-dessous
        self.in_channels = 1

        # ── Classes ───────────────────────────────────────────────────────
        self.num_classes = 50

        # ── Résolution d'entrée ───────────────────────────────────────────
        # Les images sont 128x128. YOLOX attend des multiples de 32.
        # On upscale à 160x160 pour garder un peu de marge pour le FPN.
        self.input_size = (160, 160)
        self.test_size = (160, 160)
        # Pas de multiscale training agressif sur des petites images radar
        self.multiscale_range = 1

        # ── Dataset ───────────────────────────────────────────────────────
        self.data_dir = "data/SOC_50classes_coco"
        self.train_ann = "train.json"
        self.val_ann = "test.json"
        self.train_img_dir = "images/train"
        self.val_img_dir = "images/test"

        # ── Augmentations ─────────────────────────────────────────────────
        # Augmentations légères adaptées aux images radar
        self.mosaic_prob = 0.0      # Désactivé : trop destructeur sur 128x128
        self.mixup_prob = 0.0       # Désactivé pour commencer
        self.hsv_prob = 0.0         # Inutile en grayscale
        self.flip_prob = 0.5        # Flip horizontal
        self.degrees = 10.0         # Rotation légère (±10°)
        self.translate = 0.1        # Translation légère
        self.scale = (0.8, 1.2)     # Zoom léger
        self.shear = 2.0            # Cisaillement léger
        self.enable_mixup = False

        # ── Entraînement ──────────────────────────────────────────────────
        self.max_epoch = 100
        # Batch size adapté à Colab T4 (16 Go VRAM) avec images 160x160
        self.train_batch_size = 32
        self.eval_batch_size = 32

        # Nombre de workers DataLoader
        self.data_num_workers = 2   # Réduit pour Colab

        # ── Optimiseur ────────────────────────────────────────────────────
        self.basic_lr_per_img = 0.01 / 64.0
        self.weight_decay = 5e-4
        self.momentum = 0.9
        # Warmup
        self.warmup_epochs = 5
        self.warmup_lr = 0.0
        self.min_lr_ratio = 0.05
        self.scheduler = "yoloxwarmcos"

        # ── EMA & checkpointing ───────────────────────────────────────────
        self.ema = True
        self.save_history_ckpt = False  # Économise de l'espace sur Colab

        # ── Évaluation ────────────────────────────────────────────────────
        self.eval_interval = 5       # Évaluer tous les 5 epochs
        self.test_conf = 0.01
        self.nmsthre = 0.65

        # ── Outputs ───────────────────────────────────────────────────────
        self.output_dir = "outputs"

    def get_model(self):
        """
        Override pour adapter le premier layer conv à 1 canal (grayscale).
        Par défaut YOLOX attend 3 canaux (RGB).
        """
        from yolox.models import YOLOX, YOLOXHead, YOLOPAFPN
        import torch.nn as nn

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                act=self.act,
            )

            # ── Patch : remplacer le stem conv 3→1 canal ──────────────────
            # Le stem de DarkNet est backbone.backbone.stem
            stem = backbone.backbone.stem
            old_conv = stem.conv
            new_conv = nn.Conv2d(
                in_channels=1,                  # grayscale
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Initialisation : moyenne des poids RGB pour le canal unique
            with __import__("torch").no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            stem.conv = new_conv

            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """Retourne le dataset d'entraînement custom (images .tif)."""
        from src.dataset_ import RadarCOCODataset
        from src.transform import RadarTrainTransform

        return RadarCOCODataset(
            data_dir=self.data_dir,
            ann_file=self.train_ann,
            img_dir=self.train_img_dir,
            img_size=self.input_size,
            transform=RadarTrainTransform(
                img_size=self.input_size,
                flip_prob=self.flip_prob,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
            ),
            cache=cache,
        )

    def get_eval_dataset(self, **kwargs):
        """Retourne le dataset d'évaluation custom."""
        from src.dataset_ import RadarCOCODataset
        from src.transform import RadarValTransform

        return RadarCOCODataset(
            data_dir=self.data_dir,
            ann_file=self.val_ann,
            img_dir=self.val_img_dir,
            img_size=self.test_size,
            transform=RadarValTransform(img_size=self.test_size),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        """Retourne l'évaluateur COCO standard."""
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )