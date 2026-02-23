#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# SLURM job script — Entraînement YOLOX-S (détection radar)
#
# Soumission :
#   sbatch submit_slurm.sh
#
# Vérification avant soumission :
#   sbatch --test-only submit_slurm.sh
# ══════════════════════════════════════════════════════════════════════════════

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --job-name=yolox_s_radar
#SBATCH --output=logs/slurm_%j.out       # stdout  (créer logs/ avant de soumettre)
#SBATCH --error=logs/slurm_%j.err        # stderr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # workers DataLoader (= data_num_workers * 2)
#SBATCH --gres=gpu:1                    # 1 GPU ; remplacer par gpu:a100:1 si nécessaire
#SBATCH --mem=24G
#SBATCH --time=24:00:00                 # HH:MM:SS — adapter selon le cluster
##SBATCH --partition=gpu               # décommenter et adapter si requis
##SBATCH --account=your_project        # décommenter si votre cluster demande un compte

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — adapter à votre environnement
# ══════════════════════════════════════════════════════════════════════════════

# Répertoire du projet (là où se trouve train_cluster.py)
PROJECT_DIR="$HOME/ATR_SAR"

# Chemin vers le dataset sur le stockage partagé.
# Ce dossier doit contenir directement images/ et annotations/.
DATA_DIR="/scratch/$USER/SOC_50classes_coco/SOC_50classes_coco"

# Répertoire de sortie des checkpoints (sur stockage local ou scratch)
OUTPUT_DIR="/scratch/$USER/outputs/yolox_radar"

# Environnement Python
# -- Option A : virtualenv
VENV_PATH="$HOME/.venvs/yolox"
# -- Option B : conda (décommenter les lignes conda ci-dessous)
# CONDA_ENV="yolox"

# Paramètres d'entraînement
MAX_EPOCH=100
BATCH_SIZE=32
FP16="--fp16"          # mettre "--no-fp16" pour désactiver

# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

echo "════════════════════════════════════════════════════"
echo " SLURM Job ID   : $SLURM_JOB_ID"
echo " Nœud(s)        : $SLURM_NODELIST"
echo " GPUs alloués   : ${SLURM_GPUS_ON_NODE:-$CUDA_VISIBLE_DEVICES}"
echo " Début          : $(date)"
echo "════════════════════════════════════════════════════"

# ── Charger les modules système si nécessaire ─────────────────────────────────
# Décommenter et adapter selon votre cluster :
# module purge
# module load cuda/12.1 python/3.10

# ── Activer l'environnement Python ────────────────────────────────────────────
# Option A : virtualenv
source "$VENV_PATH/bin/activate"

# Option B : conda (décommenter)
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "$CONDA_ENV"

# ── Aller dans le répertoire du projet ───────────────────────────────────────
cd "$PROJECT_DIR" || { echo "Répertoire non trouvé : $PROJECT_DIR"; exit 1; }

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
python train_cluster.py \
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
