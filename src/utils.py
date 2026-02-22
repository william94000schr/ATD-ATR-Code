#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaires pour l'intégration MLflow et le logging d'entraînement YOLOX.

Usage :
    from src.utils import MLflowLogger

    logger = MLflowLogger(experiment_name="yolox_radar")
    logger.start_run(run_name="yolox_s_100ep")
    logger.log_params({"epochs": 100, "batch_size": 32})
    logger.log_metrics({"loss": 0.42, "mAP": 0.65}, step=10)
    logger.log_artifact("outputs/best_ckpt.pth")
    logger.end_run()
"""

import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch


# ── Configuration par défaut ──────────────────────────────────────────────────

DEFAULT_TRACKING_URI = "outputs/mlruns"
DEFAULT_EXPERIMENT = "yolox_radar_detection"


# ── Logger principal ──────────────────────────────────────────────────────────

class MLflowLogger:
    """
    Wrapper MLflow pour l'entraînement YOLOX.

    Simplifie le logging des hyperparamètres, métriques, artefacts
    et checkpoints PyTorch.

    Args:
        experiment_name : Nom de l'expérience MLflow
        tracking_uri    : Chemin local vers le dossier mlruns
                          (lancer 'mlflow ui --backend-store-uri outputs/mlruns')
        tags            : Tags additionnels attachés au run
    """

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT,
        tracking_uri: str = DEFAULT_TRACKING_URI,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self._run = None

        # Création du dossier de tracking si nécessaire
        Path(tracking_uri).mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    # ── Cycle de vie du run ───────────────────────────────────────────────

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Démarre un run MLflow. Ajoute des tags système automatiquement."""
        system_tags = {
            "host": socket.gethostname(),
            "python.version": _get_python_version(),
            **self.tags,
        }
        self._run = mlflow.start_run(run_name=run_name, tags=system_tags)
        print(
            f"[MLflow] Run démarré : {self._run.info.run_id}\n"
            f"[MLflow] UI → mlflow ui --backend-store-uri {self.tracking_uri}"
        )

    def end_run(self) -> None:
        """Termine proprement le run MLflow."""
        if self._run is not None:
            mlflow.end_run()
            print(f"[MLflow] Run terminé : {self._run.info.run_id}")
            self._run = None

    @property
    def run_id(self) -> Optional[str]:
        return self._run.info.run_id if self._run else None

    # ── Logging ──────────────────────────────────────────────────────────

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log un dictionnaire d'hyperparamètres."""
        # MLflow limite les params à 100 par appel
        items = list(params.items())
        for i in range(0, len(items), 100):
            mlflow.log_params(dict(items[i : i + 100]))

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log une métrique scalaire."""
        mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log un dictionnaire de métriques."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Upload un fichier (checkpoint, config, image...) dans le run."""
        if Path(local_path).exists():
            mlflow.log_artifact(local_path, artifact_path)
        else:
            print(f"[MLflow] Artefact introuvable, ignoré : {local_path}")

    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log un modèle PyTorch dans MLflow."""
        mlflow.pytorch.log_model(model, artifact_path)

    # ── Extraction des params depuis la config Exp ────────────────────────

    def log_exp_config(self, exp) -> None:
        """
        Extrait et log les hyperparamètres pertinents depuis un objet Exp YOLOX.
        """
        params = {
            "exp_name": exp.exp_name,
            "num_classes": exp.num_classes,
            "depth": exp.depth,
            "width": exp.width,
            "input_size": str(exp.input_size),
            "max_epoch": exp.max_epoch,
            "train_batch_size": exp.train_batch_size,
            "basic_lr_per_img": exp.basic_lr_per_img,
            "weight_decay": exp.weight_decay,
            "warmup_epochs": exp.warmup_epochs,
            "mosaic_prob": exp.mosaic_prob,
            "mixup_prob": exp.mixup_prob,
            "flip_prob": exp.flip_prob,
            "degrees": exp.degrees,
            "translate": exp.translate,
            "scale": str(exp.scale),
            "shear": exp.shear,
            "ema": exp.ema,
            "scheduler": exp.scheduler,
        }
        self.log_params(params)


# ── Hook YOLOX → MLflow ───────────────────────────────────────────────────────

class MLflowTrainerHook:
    """
    Hook à brancher sur le Trainer YOLOX pour logger automatiquement
    les métriques à chaque epoch.

    Usage dans train.py :
        hook = MLflowTrainerHook(logger)
        trainer.after_epoch_hooks.append(hook)
    """

    def __init__(self, mlflow_logger: MLflowLogger):
        self.logger = mlflow_logger

    def __call__(self, trainer) -> None:
        """Appelé par le trainer YOLOX après chaque epoch."""
        epoch = trainer.epoch
        meter = trainer.meter

        metrics = {}

        # Métriques de loss (disponibles dans le meter YOLOX)
        for key in ("total_loss", "iou_loss", "conf_loss", "cls_loss", "l1_loss"):
            if key in meter:
                metrics[f"train/{key}"] = meter[key].global_avg

        # mAP si disponible (après évaluation)
        if hasattr(trainer, "ap50_95"):
            metrics["val/mAP_50_95"] = trainer.ap50_95
        if hasattr(trainer, "ap50"):
            metrics["val/mAP_50"] = trainer.ap50

        if metrics:
            self.logger.log_metrics(metrics, step=epoch)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_python_version() -> str:
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def setup_output_dirs(output_dir: str = "outputs") -> Dict[str, Path]:
    """
    Crée et retourne les dossiers de sortie standards.

    Structure créée :
        outputs/
        ├── checkpoints/
        ├── mlruns/
        └── logs/
    """
    base = Path(output_dir)
    dirs = {
        "checkpoints": base / "checkpoints",
        "mlruns": base / "mlruns",
        "logs": base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def print_dataset_stats(dataset, split: str = "train") -> None:
    """Affiche des statistiques rapides sur un RadarCOCODataset."""
    n_images = len(dataset)
    n_boxes = sum(len(a["boxes"]) for a in dataset.annotations)
    print(f"[Dataset/{split}]")
    print(f"  Images   : {n_images}")
    print(f"  Boîtes   : {n_boxes}")
    print(f"  Moy/img  : {n_boxes / max(n_images, 1):.2f}")
    print(f"  Classes  : {len(dataset.class_ids)}")