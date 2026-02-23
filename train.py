#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point d'entrée pour l'entraînement YOLOX-S sur le dataset radar.

Usage :
    python train.py --config config/yolo_s.py
    python train.py --config config/yolo_s.py --max-epoch 50 --batch-size 32
    python train.py --config config/yolo_s.py --resume --ckpt outputs/checkpoints/last_epoch_ckpt.pth
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement YOLOX-S pour la détection de cibles radar"
    )
    parser.add_argument("--config", type=str, default="config/yolox_s_radar.py",
                        help="Chemin vers le fichier de configuration Exp YOLOX")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Reprendre depuis --ckpt")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint .pth à charger")
    parser.add_argument("--max-epoch", type=int, default=None,
                        help="Override du nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override du batch size")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Nom du run (informatif)")
    parser.add_argument("--devices", type=int, default=1,
                        help="Nombre de GPUs (1 pour Colab)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Mixed precision FP16")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="Désactiver FP16")
    parser.add_argument("--cache", type=str, default=None,
                        help="Cache images (None, 'ram', 'disk')")

    # Arguments requis par le Trainer YOLOX interne
    parser.add_argument("--experiment-name", type=str, default="yolox_radar_detection")
    parser.add_argument("--occupy", action="store_true", default=False,
                        help="Pré-allouer toute la VRAM GPU")
    parser.add_argument("--logger", type=str, default=None,
                        help="Logger backend (tensorboard, wandb, ou None)")
    parser.add_argument("--start-epoch", type=int, default=None,
                        help="Epoch de départ pour le resume")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None,
                        help="Overrides de config supplémentaires (clé=valeur)")

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

    # ── Dossiers de sortie ───────────────────────────────────────────────────
    output_dir = Path("outputs")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Setup] Checkpoints → {ckpt_dir}/")

    # ── Chargement de la config ──────────────────────────────────────────────
    print(f"[Config] {args.config}")
    exp = load_exp(args.config)

    if args.max_epoch is not None:
        exp.max_epoch = args.max_epoch
    if args.batch_size is not None:
        exp.train_batch_size = args.batch_size

    # Le Trainer YOLOX lit args.batch_size directement pour get_data_loader
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

    # ── Statistiques dataset ─────────────────────────────────────────────────
    print("\n[Dataset]")
    train_ds = exp.get_dataset()
    val_ds = exp.get_eval_dataset()
    for ds, split in [(train_ds, "train"), (val_ds, "val")]:
        n_img = len(ds)
        n_boxes = sum(len(a["boxes"]) for a in ds.annotations)
        print(f"  {split:5s}: {n_img} images, {n_boxes} boxes ({n_boxes / max(n_img, 1):.1f}/img), {len(ds.class_ids)} classes")

    # ── Modèle ───────────────────────────────────────────────────────────────
    print("\n[Model]")
    model = exp.get_model()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres totaux      : {n_params:,}")
    print(f"  Paramètres entraînables: {n_trainable:,}")

    # ── Entraînement ─────────────────────────────────────────────────────────
    print("\n[Training] Démarrage...")
    from yolox.core import Trainer
    trainer = Trainer(exp, args)
    trainer.train()

    # ── Résumé final ─────────────────────────────────────────────────────────
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
