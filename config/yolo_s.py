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
from pathlib import Path

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        self.exp_name = "yolox_s_radar_50cls"
        self.depth = 0.33
        self.width = 0.50
        self.act = "silu"
        self.in_channels = 1
        self.num_classes = 50
        self.input_size = (160, 160)
        self.test_size = (160, 160)
        self.multiscale_range = 1

        self.data_dir = "data/SOC_50classes_coco/SOC_50classes_coco"
        self.train_ann = "train.json"
        self.val_ann = "test.json"
        self.train_img_dir = "images/train"
        self.val_img_dir = "images/test"

        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.8, 1.2)
        self.shear = 2.0
        self.enable_mixup = False

        self.max_epoch = 100
        self.train_batch_size = 32
        self.eval_batch_size = 32
        self.data_num_workers = 2

        self.basic_lr_per_img = 0.01 / 64.0
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.warmup_epochs = 5
        self.warmup_lr = 0.0
        self.min_lr_ratio = 0.05
        self.scheduler = "yoloxwarmcos"

        self.ema = True
        self.save_history_ckpt = False
        self.eval_interval = 5
        self.test_conf = 0.01
        self.nmsthre = 0.65
        self.output_dir = "outputs"

    def get_model(self):
        from yolox.models import YOLOX, YOLOXHead, YOLOPAFPN
        import torch.nn as nn

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)

            # BaseConv contient une nn.Conv2d dans .conv
            old_conv = backbone.backbone.stem.conv  # c'est le nn.Conv2d réel
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with __import__("torch").no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            # Remplacer le nn.Conv2d interne du BaseConv
            backbone.backbone.stem.conv = new_conv

            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_dataset(self, cache=False, cache_type="ram"):
        from src.dataset import SAR_ATR_Dataset
        from src.transform import RadarTrainTransform

        return SAR_ATR_Dataset(
            root=str(Path(self.data_dir) / self.train_img_dir),
            annFile=str(Path(self.data_dir) / "annotations" / self.train_ann),
            transforms=RadarTrainTransform(
                img_size=self.input_size,
                flip_prob=self.flip_prob,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
            ),
            img_size=self.input_size,
        )

    def get_eval_dataset(self, **kwargs):
        from src.dataset import SAR_ATR_Dataset
        from src.transform import RadarValTransform

        return SAR_ATR_Dataset(
            root=str(Path(self.data_dir) / self.val_img_dir),
            annFile=str(Path(self.data_dir) / "annotations" / self.val_ann),
            transforms=RadarValTransform(img_size=self.test_size),
            img_size=self.test_size,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )