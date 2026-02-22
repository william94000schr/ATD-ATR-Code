#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évaluation du modèle YOLOX entraîné sur le dataset radar.

Calcule les métriques COCO (mAP@0.5, mAP@0.5:0.95) sur le set de test
et log les résultats dans MLflow.

Usage :
    python evaluate.py --ckpt outputs/checkpoints/best_ckpt.pth

    # Avec un seuil de confiance personnalisé
    python evaluate.py --ckpt outputs/checkpoints/best_ckpt.pth --conf 0.25

    # Logguer dans un run MLflow existant
    python evaluate.py --ckpt outputs/checkpoints/best_ckpt.pth --run-id <RUN_ID>
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.utils import MLflowLogger, setup_output_dirs, print_dataset_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Évaluation YOLOX sur le dataset radar test"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/yolox_s_radar.py",
        help="Chemin vers le fichier de configuration Exp",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Chemin vers le checkpoint .pth à évaluer",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.01,
        help="Seuil de confiance pour la détection (défaut: 0.01 pour mAP COCO)",
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=0.65,
        help="Seuil NMS IoU",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size pour l'inférence",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="ID d'un run MLflow existant pour y logger les résultats",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="yolox_radar_detection",
        help="Nom de l'expérience MLflow",
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
    dirs = setup_output_dirs("outputs")

    # ── Config ────────────────────────────────────────────────────────────
    print(f"[Config] Chargement depuis : {args.config}")
    exp = load_exp(args.config)
    exp.test_conf = args.conf
    exp.nmsthre = args.nms

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Modèle ────────────────────────────────────────────────────────────
    print(f"[Modèle] Chargement du checkpoint : {args.ckpt}")
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location=device)

    # Les checkpoints YOLOX peuvent contenir 'model' ou 'state_dict'
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("[Modèle] Chargé avec succès.")

    # ── Dataset test ──────────────────────────────────────────────────────
    val_dataset = exp.get_eval_dataset()
    print_dataset_stats(val_dataset, split="test")

    # ── Évaluateur COCO ───────────────────────────────────────────────────
    print("\n[Evaluation] Calcul des métriques COCO...")
    evaluator = exp.get_evaluator(
        batch_size=args.batch_size,
        is_distributed=False,
    )

    with torch.no_grad():
        ap50_95, ap50, summary = evaluator.evaluate(
            model,
            is_distributed=False,
            half=device.type == "cuda",
        )

    print("\n" + "=" * 60)
    print(f"  mAP @ IoU=0.50:0.95 : {ap50_95:.4f}")
    print(f"  mAP @ IoU=0.50      : {ap50:.4f}")
    print("=" * 60)
    if summary:
        print(summary)

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow_logger = MLflowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=str(dirs["mlruns"]),
    )

    import mlflow
    if args.run_id:
        # Logger dans un run existant (celui de l'entraînement)
        with mlflow.start_run(run_id=args.run_id):
            mlflow_logger._run = mlflow.active_run()
            mlflow_logger.log_metrics(
                {
                    "eval/mAP_50_95": ap50_95,
                    "eval/mAP_50": ap50,
                    "eval/conf_threshold": args.conf,
                    "eval/nms_threshold": args.nms,
                }
            )
    else:
        mlflow_logger.start_run(run_name=f"eval_{Path(args.ckpt).stem}")
        mlflow_logger.log_params(
            {
                "ckpt": args.ckpt,
                "conf_threshold": args.conf,
                "nms_threshold": args.nms,
                "num_classes": exp.num_classes,
            }
        )
        mlflow_logger.log_metrics(
            {
                "eval/mAP_50_95": ap50_95,
                "eval/mAP_50": ap50,
            }
        )
        mlflow_logger.log_artifact(args.ckpt, artifact_path="checkpoint")
        mlflow_logger.end_run()

    print(f"\n[MLflow] Résultats enregistrés.")
    print(f"[MLflow] Visualiser : mlflow ui --backend-store-uri {dirs['mlruns']}")


if __name__ == "__main__":
    main()