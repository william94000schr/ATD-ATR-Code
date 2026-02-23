#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point d'entrée pour l'entraînement YOLOX sur le dataset radar.

Usage :
    # Entraînement standard
    python train.py

    # Reprendre depuis un checkpoint
    python train.py --resume --ckpt outputs/checkpoints/last_epoch_ckpt.pth

    # Entraînement avec un nombre d'epochs différent
    python train.py --max-epoch 50

    # Nommer le run MLflow
    python train.py --run-name "experience_v2_lr0.01"
"""

import argparse
import sys
from pathlib import Path

# ── Ajout du projet au PYTHONPATH ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import MLflowLogger, setup_output_dirs, print_dataset_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement YOLOX-S pour la détection de cibles radar"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/yolox_s_radar.py",
        help="Chemin vers le fichier de configuration Exp YOLOX",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprendre l'entraînement depuis --ckpt",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Chemin vers un checkpoint .pth (resume ou fine-tuning)",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="Override du nombre d'epochs défini dans la config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override du batch size défini dans la config",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nom du run MLflow",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Nombre de GPUs à utiliser (1 pour Colab)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Entraînement en mixed precision FP16 (recommandé sur T4)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Désactiver le FP16",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="yolox_radar_detection",
        help="Nom de l'expérience MLflow",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Cache images en RAM ou disque (None, 'ram', 'disk')",
    )
    return parser.parse_args()


def load_exp(config_path: str):
    """Charge dynamiquement le fichier Exp YOLOX."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Exp()


def main():
    args = parse_args()

    # ── Dossiers de sortie ────────────────────────────────────────────────
    dirs = setup_output_dirs("outputs")
    print(f"[Setup] Dossiers de sortie créés dans : outputs/")

    # ── Chargement de la config ───────────────────────────────────────────
    print(f"[Config] Chargement depuis : {args.config}")
    exp = load_exp(args.config)

    # Overrides CLI
    if args.max_epoch is not None:
        exp.max_epoch = args.max_epoch
        print(f"[Config] max_epoch overridé → {exp.max_epoch}")
    if args.batch_size is not None:
        exp.train_batch_size = args.batch_size
        print(f"[Config] batch_size overridé → {exp.train_batch_size}")

    # Rediriger les outputs YOLOX vers notre dossier
    exp.output_dir = str(dirs["checkpoints"])

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow_logger = MLflowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=str(dirs["mlruns"]),
    )
    run_name = args.run_name or f"{exp.exp_name}_ep{exp.max_epoch}"
    mlflow_logger.start_run(run_name=run_name)
    mlflow_logger.log_exp_config(exp)

    # Log des overrides CLI
    mlflow_logger.log_params(
        {
            "fp16": args.fp16,
            "devices": args.devices,
            "resume": args.resume,
            "ckpt": args.ckpt or "none",
        }
    )

    # ── Statistiques dataset ──────────────────────────────────────────────
    print("\n[Dataset] Chargement des statistiques...")
    train_dataset = exp.get_dataset()
    val_dataset = exp.get_eval_dataset()
    print_dataset_stats(train_dataset, split="train")
    print_dataset_stats(val_dataset, split="test")

    mlflow_logger.log_params(
        {
            "dataset/train_images": len(train_dataset),
            "dataset/val_images": len(val_dataset),
        }
    )

    # ── Modèle ────────────────────────────────────────────────────────────
    print("\n[Modèle] Construction du modèle YOLOX-S...")
    model = exp.get_model()

    n_params = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres totaux      : {n_params:,}")
    print(f"  Paramètres entraînables: {n_params_trainable:,}")
    mlflow_logger.log_params(
        {
            "model/total_params": n_params,
            "model/trainable_params": n_params_trainable,
        }
    )

    # ── Log du fichier de config ──────────────────────────────────────────
    mlflow_logger.log_artifact(args.config, artifact_path="config")

    # ── Lancement de l'entraînement YOLOX ────────────────────────────────
    print("\n[Training] Démarrage de l'entraînement...")
    try:
        from yolox.core import Trainer

        trainer = Trainer(exp, args)

        # Patch : injection du logger MLflow dans le trainer
        _patch_trainer_with_mlflow(trainer, mlflow_logger)

        trainer.train()

    except Exception as e:
        mlflow_logger.log_params({"status": "FAILED", "error": str(e)[:200]})
        mlflow_logger.end_run()
        raise

    # ── Sauvegarde du meilleur checkpoint dans MLflow ─────────────────────
    best_ckpt = dirs["checkpoints"] / exp.exp_name / "best_ckpt.pth"
    if best_ckpt.exists():
        mlflow_logger.log_artifact(str(best_ckpt), artifact_path="checkpoints")
        print(f"[MLflow] Meilleur checkpoint uploadé : {best_ckpt}")

    mlflow_logger.end_run()
    print("\n[Done] Entraînement terminé.")
    print(f"[MLflow] Lancer : mlflow ui --backend-store-uri {dirs['mlruns']}")


def _patch_trainer_with_mlflow(trainer, mlflow_logger: MLflowLogger) -> None:
    """
    Injecte le logging MLflow dans le Trainer YOLOX en monkey-patching
    la méthode after_epoch.

    On wrappe after_epoch pour logger les métriques disponibles dans
    trainer.meter après chaque epoch.
    """
    original_after_epoch = trainer.after_epoch

    def patched_after_epoch():
        original_after_epoch()

        epoch = trainer.epoch
        meter = getattr(trainer, "meter", {})
        metrics = {}

        for key in ("total_loss", "iou_loss", "conf_loss", "cls_loss", "l1_loss"):
            if key in meter and hasattr(meter[key], "global_avg"):
                metrics[f"train/{key}"] = meter[key].global_avg

        if hasattr(trainer, "ap50_95") and trainer.ap50_95 is not None:
            metrics["val/mAP_50_95"] = trainer.ap50_95
        if hasattr(trainer, "ap50") and trainer.ap50 is not None:
            metrics["val/mAP_50"] = trainer.ap50

        if metrics:
            mlflow_logger.log_metrics(metrics, step=epoch)

    trainer.after_epoch = patched_after_epoch


if __name__ == "__main__":
    main()