#!/bin/bash
#SBATCH --job-name=fanout_buildidx
#SBATCH --partition=ampere
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/hpc/share/kangdo/mas-latency/results/fanout/buildidx-%j.out
set -u

BASE="/nfs/hpc/share/kangdo/personal_ma"
REPO="/nfs/hpc/share/kangdo/mas-latency"
LG_VENV="$BASE/envs/langgraph"

export TMPDIR="/scratch/kangdo/tmp"; mkdir -p "$TMPDIR"

export FANOUT_BACKEND="${FANOUT_BACKEND:-bm25s}"
export FANOUT_CORPUS="${FANOUT_CORPUS:-$BASE/data/fanout/corpus.jsonl}"
export FANOUT_DISTRACTORS="${FANOUT_DISTRACTORS:-$BASE/data/fanout/distractors.jsonl}"
export FANOUT_INDEX="${FANOUT_INDEX:-$BASE/data/fanout/fanout_index_bm25s_distract.pkl}"

echo "node=$(hostname)  time=$(date)"
echo "backend=$FANOUT_BACKEND  distractors=$FANOUT_DISTRACTORS  index=$FANOUT_INDEX"
free -g | head -2

"$LG_VENV/bin/python" - <<'PY'
import sys, time
sys.path.insert(0, "/nfs/hpc/share/kangdo/mas-latency")
import fanout_mas.tools_fanout as T
t0 = time.time()
n = T.build_corpus_index(force=True)
print(f"[build] 완료: {n} 청크, {time.time()-t0:.0f}s -> {T._default_pickle()}", flush=True)
PY
echo "done  time=$(date)"
ls -lh "$FANOUT_INDEX"
