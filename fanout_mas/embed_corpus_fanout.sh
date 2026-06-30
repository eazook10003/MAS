#!/bin/bash
#SBATCH --job-name=fanout_embed
#SBATCH --partition=ampere
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/hpc/share/kangdo/mas-latency/results/fanout/embed-%j.out
set -u

BASE="/nfs/hpc/share/kangdo/personal_ma"
REPO="/nfs/hpc/share/kangdo/mas-latency"
LG_VENV="$BASE/envs/langgraph"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="/scratch/kangdo/hf_home"; export TMPDIR="/scratch/kangdo/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"
export HF_HUB_OFFLINE=0

export FANOUT_CORPUS="${FANOUT_CORPUS:-$BASE/data/fanout/corpus.jsonl}"
export FANOUT_DISTRACTORS="${FANOUT_DISTRACTORS:-$BASE/data/fanout/distractors.jsonl}"
export FANOUT_EMB="${FANOUT_EMB:-$BASE/data/fanout/fanout_dense_emb.npy}"
export FANOUT_EMB_DOCS="${FANOUT_EMB_DOCS:-$BASE/data/fanout/fanout_dense_docs.pkl}"

export FANOUT_EMB_MODEL="${FANOUT_EMB_MODEL:-$BASE/data/fanout/emb_model}"
[ -d "$FANOUT_EMB_MODEL" ] || HF_HUB_DISABLE_XET=1 "$LG_VENV/bin/python" -c \
  "from huggingface_hub import snapshot_download as d; d('sentence-transformers/all-MiniLM-L6-v2', local_dir='$FANOUT_EMB_MODEL')"

echo "node=$(hostname)  time=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
free -g | head -2

"$LG_VENV/bin/python" "$REPO/fanout_mas/embed_corpus.py"

echo "done  time=$(date)"
ls -lh "$FANOUT_EMB" "$FANOUT_EMB_DOCS" 2>/dev/null
