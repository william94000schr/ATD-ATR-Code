#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# SLURM job script — Entraînement YOLOX-S (détection radar)
# Cluster : PANDO — ISAE-SUPAERO
#
# Soumission :
#   sbatch submit_slurm.sh
#
# Vérification avant soumission :
#   sbatch --test-only submit_slurm.sh
# ══════════════════════════════════════════════════════════════════════════════

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --job-name=yolox_s_radar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12              # PANDO : 24 cœurs/nœud (2x Xeon Gold 6126 dodeca-core)
                                        # 12 workers pour le DataLoader, adapté à vos images 128x128
#SBATCH --gres=gpu:1                    # 1 GPU sur gpu[01-02]
                                        # gpu01 : Tesla V100 32Go / gpu02 : A100 40Go
                                        # Pour cibler l'A100 : --gres=gpu:a100:1
#SBATCH --mem=64G                       # gpu[01-02] disposent de 192Go de RAM
#SBATCH --time=24:00:00                 # HH:MM:SS
#SBATCH --partition=gpu                 # Partition GPU de PANDO
#SBATCH --begin=now
#SBATCH --mail-user=mael.dacher@student.isae-supaero.fr
#SBATCH --mail-type=FAIL,END,SUCCESS
#SBATCH -o logs/slurm.%j.out
#SBATCH -e logs/slurm.%j.err

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — adapter à votre environnement
# ══════════════════════════════════════════════════════════════════════════════

# Répertoire du projet (là où se trouve train_cluster.py)
PROJECT_DIR="/scratch/students/$HOME/ATR_SAR"

# Chemin vers le dataset sur le stockage rapide BeeGFS (préférer /scratch/ à /home/)
DATA_DIR="/scratch/student/$USER/SOC_40classes_coco/SOC_40classes_coco"

# Répertoire de sortie des checkpoints
OUTPUT_DIR="/scratch/students/$USER/outputs/yolox_radar"

# Paramètres d'entraînement
MAX_EPOCH=30
BATCH_SIZE=32
FP16="--fp16"          # mettre "--no-fp16" pour désactiver

# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Proxy ISAE (requis pour tout accès réseau externe depuis PANDO) ───────────
export https_proxy=http://proxy.isae.fr:3128
export http_proxy=http://proxy.isae.fr:3128

echo "════════════════════════════════════════════════════"
echo " SLURM Job ID   : $SLURM_JOB_ID"
echo " Répertoire     : ${SLURM_SUBMIT_DIR}"
echo " Nœud(s)        : $SLURM_NODELIST"
echo " GPUs alloués   : ${SLURM_GPUS_ON_NODE:-$CUDA_VISIBLE_DEVICES}"
echo " Début          : $(date)"
echo "════════════════════════════════════════════════════"

# Se placer dans le répertoire de soumission (bonne pratique PANDO)
cd "${SLURM_SUBMIT_DIR}"

# ── Charger les modules système ───────────────────────────────────────────────
module purge
module load gcc/12.2.0
# Décommenter selon les modules disponibles sur PANDO (voir: module avail) :
# module load cuda/12.1
# module load python/3.10

# ── Activer uv et installer les dépendances ───────────────────────────────────
source "$HOME/.local/bin/env"           # initialise uv (installé en user sur PANDO)
cd "$PROJECT_DIR" || { echo "Répertoire non trouvé : $PROJECT_DIR"; exit 1; }

# Installe toutes les dépendances verrouillées dans uv.lock (torch, numpy, etc.)
uv sync --no-dev

# yolox ne peut pas être dans pyproject.toml car son setup.py importe torch
# lors de la résolution des dépendances (avant que torch soit installé).
# On l'installe séparément sans ses propres deps (déjà listées dans pyproject.toml).
uv pip install --no-deps yolox==0.3.0

# ── Créer les répertoires nécessaires ────────────────────────────────────────
mkdir -p logs "$OUTPUT_DIR"

# ── Vérifier le dataset ───────────────────────────────────────────────────────
if [ ! -d "$DATA_DIR" ]; then
    echo "Erreur : dataset introuvable → $DATA_DIR"
    exit 1
fi
echo "Dataset : $DATA_DIR"
echo "  Train : $(ls "$DATA_DIR/images/train" | wc -l) images"
echo "  Test  : $(ls "$DATA_DIR/images/test"  | wc -l) images"

# ── Lancement de l'entraînement ───────────────────────────────────────────────
uv run python train_cluster.py \
    --config     config/yolo_s.py \
    --data-dir   "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-epoch  "$MAX_EPOCH" \
    --batch-size "$BATCH_SIZE" \
    $FP16

echo "════════════════════════════════════════════════════"
echo " Fin : $(date)"
echo " Checkpoints dans : $OUTPUT_DIR/checkpoints/"
echo "════════════════════════════════════════════════════"