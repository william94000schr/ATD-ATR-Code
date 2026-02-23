#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraînement YOLOX-S — environnement cluster HPC / SLURM.

Équivalent de train.py, avec gestion des chemins absolus propre à un cluster
(dataset sur stockage partagé, checkpoints sur stockage local/scratch).

Usage direct :
    python train_cluster.py \\
        --data-dir  /scratch/$USER/SOC_50classes_coco/SOC_50classes_coco \\
        --output-dir /scratch/$USER/outputs/yolox_radar

Soumission SLURM :
    sbatch submit_slurm.sh

Resume depuis un checkpoint :
    python train_cluster.py --resume --ckpt /path/to/last_epoch_ckpt.pth ...
"""

import importlib.util
import os
import sys
from pathlib import Path

import argparse

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


# ── Valeurs par défaut (adapter à votre cluster) ──────────────────────────────
_DEFAULT_DATA_DIR   = str(PROJECT_ROOT / "data" / "SOC_50classes_coco" / "SOC_50classes_coco")
_DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs")
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement YOLOX-S sur cluster SLURM"
    )

    # ── Chemins ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--config", type=str, default="config/yolo_s.py",
        help="Fichier de configuration Exp YOLOX",
    )
    parser.add_argument(
        "--data-dir", type=str, default=_DEFAULT_DATA_DIR,
        help=(
            "Chemin absolu vers le dataset (dossier contenant images/ et annotations/). "
            "Exemple : /scratch/$USER/SOC_50classes_coco/SOC_50classes_coco"
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie des checkpoints (chemin absolu recommandé)",
    )

    # ── Entraînement ──────────────────────────────────────────────────────────
    parser.add_argument("--max-epoch",  type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--run-name",   type=str,   default=None)
    parser.add_argument("--devices",    type=int,   default=1)
    parser.add_argument("--fp16",       action="store_true", default=True)
    parser.add_argument("--no-fp16",    dest="fp16", action="store_false")
    parser.add_argument("--cache",      type=str,   default=None,
                        help="Cache images en RAM ou sur disque ('ram' | 'disk')")

    # ── Resume ────────────────────────────────────────────────────────────────
    parser.add_argument("--resume",      action="store_true", default=False)
    parser.add_argument("--ckpt",        type=str, default=None,
                        help="Checkpoint .pth à charger")
    parser.add_argument("--start-epoch", type=int, default=None)

    # ── Arguments internes YOLOX Trainer ──────────────────────────────────────
    parser.add_argument("--experiment-name", type=str, default="yolox_radar_detection")
    parser.add_argument("--occupy",          action="store_true", default=False)
    parser.add_argument("--logger",          type=str, default="tensorboard")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)

    return parser.parse_args()


def load_exp(config_path: str):
    spec   = importlib.util.spec_from_file_location("exp_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Exp()


def log_env():
    """Affiche les informations d'environnement SLURM et GPU."""
    import torch

    slurm_keys = [
        "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NODELIST",
        "SLURM_NTASKS", "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS_ON_NODE", "CUDA_VISIBLE_DEVICES",
    ]
    in_slurm = "SLURM_JOB_ID" in os.environ

    print("[Environnement]")
    if in_slurm:
        for k in slurm_keys:
            v = os.environ.get(k, "—")
            print(f"  {k:28s} = {v}")
    else:
        print("  (exécution hors SLURM)")

    print(f"\n[GPU]")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i} : {props.name}  ({props.total_memory / 1e9:.1f} Go VRAM)")
    else:
        print("  Aucun GPU CUDA détecté")
    print(f"  PyTorch : {torch.__version__}")


def main():
    args = parse_args()

    log_env()

    # ── Répertoires de sortie ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir).resolve()
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Setup]")
    print(f"  Checkpoints → {ckpt_dir}/")

    # ── Chargement de la config ───────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    print(f"\n[Config] {config_path}")

    exp = load_exp(str(config_path))

    # Overrides depuis la ligne de commande
    exp.data_dir = args.data_dir
    if args.max_epoch  is not None:
        exp.max_epoch        = args.max_epoch
    if args.batch_size is not None:
        exp.train_batch_size = args.batch_size

    # Le Trainer YOLOX lit args.batch_size pour instancier le DataLoader
    if args.batch_size is None:
        args.batch_size = exp.train_batch_size

    exp.output_dir = str(ckpt_dir)

    run_name = args.run_name or f"{exp.exp_name}_ep{exp.max_epoch}_bs{exp.train_batch_size}"

    print(f"  run_name   : {run_name}")
    print(f"  max_epoch  : {exp.max_epoch}")
    print(f"  batch_size : {exp.train_batch_size}")
    print(f"  fp16       : {args.fp16}")
    print(f"  input_size : {exp.input_size}")
    print(f"  num_classes: {exp.num_classes}")
    print(f"  data_dir   : {exp.data_dir}")
    print(f"  output_dir : {exp.output_dir}")

    # ── Vérification du dataset ───────────────────────────────────────────────
    data_path = Path(exp.data_dir)
    if not data_path.exists():
        sys.exit(
            f"\n[Erreur] Dataset introuvable : {data_path}\n"
            f"Vérifiez --data-dir ou montez le stockage partagé."
        )

    # ── Statistiques dataset ──────────────────────────────────────────────────
    print("\n[Dataset]")
    train_ds = exp.get_dataset()
    val_ds   = exp.get_eval_dataset()
    for ds, split in [(train_ds, "train"), (val_ds, "val")]:
        n_img   = len(ds)
        n_boxes = sum(len(a["boxes"]) for a in ds.annotations)
        print(f"  {split:5s}: {n_img} images, {n_boxes} boîtes "
              f"({n_boxes / max(n_img, 1):.1f}/img), {len(ds.class_ids)} classes")

    # ── Modèle ────────────────────────────────────────────────────────────────
    print("\n[Model]")
    model      = exp.get_model()
    n_params   = sum(p.numel() for p in model.parameters())
    n_train    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres totaux       : {n_params:,}")
    print(f"  Paramètres entraînables : {n_train:,}")

    # ── Entraînement ─────────────────────────────────────────────────────────
    print("\n[Training] Démarrage...")
    from yolox.core import Trainer
    trainer = Trainer(exp, args)
    trainer.train()

    # ── Résumé ────────────────────────────────────────────────────────────────
    best_ckpt = ckpt_dir / exp.exp_name / "best_ckpt.pth"
    last_ckpt = ckpt_dir / exp.exp_name / "last_epoch_ckpt.pth"
    print("\n[Done] Entraînement terminé.")
    print(f"  Checkpoints : {ckpt_dir / exp.exp_name}/")
    if best_ckpt.exists():
        print(f"  Meilleur    : {best_ckpt}")
    if last_ckpt.exists():
        print(f"  Dernier     : {last_ckpt}")


if __name__ == "__main__":
    main()
