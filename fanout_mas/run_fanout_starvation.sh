#!/bin/bash
#SBATCH --job-name=fanout_starv
#SBATCH --partition=ampere
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/hpc/share/kangdo/mas-latency/results/fanout_starvation/slurm-%j.out
set -u

BASE="/nfs/hpc/share/kangdo/personal_ma"
REPO="/nfs/hpc/share/kangdo/mas-latency"
VLLM_VENV="$BASE/envs/vllm"; LG_VENV="$BASE/envs/langgraph"
MODEL_NAME="Qwen2.5-7B-Instruct"; MODEL_DIR="$BASE/models/$MODEL_NAME"
PORT="${PORT:-$(( 8000 + (${SLURM_JOB_ID:-$RANDOM} % 1000) ))}"

MODE="${MODE:-shared}"
VLLM_CORES_N="${VLLM_CORES_N:-1}"
N_QUESTIONS="${N_QUESTIONS:-20}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-0}"
RESEARCHER_TOOL_CAP="${RESEARCHER_TOOL_CAP:-4}"
FANOUT_BACKEND="${FANOUT_BACKEND:-bm25s}"
FANOUT_CORPUS="${FANOUT_CORPUS:-$BASE/data/fanout/corpus.jsonl}"
FANOUT_DISTRACTORS="${FANOUT_DISTRACTORS-$BASE/data/fanout/distractors.jsonl}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
EAGER_FLAG=""; [ "$ENFORCE_EAGER" = "1" ] && EAGER_FLAG="--enforce-eager"

export CUDA_HOME="/usr/local/apps/cuda/13.0"
export CPATH="/usr/local/apps/python/3.12/include/python3.12"
export TMPDIR="/scratch/kangdo/tmp"; export HF_HOME="/scratch/kangdo/hf_home"
export VLLM_CACHE_ROOT="/scratch/kangdo/vllm_cache"
export TORCHINDUCTOR_CACHE_DIR="/scratch/kangdo/inductor_cache"
export TRITON_CACHE_DIR="/scratch/kangdo/triton_cache"
mkdir -p "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="$VLLM_VENV/bin:$PATH"

expand_cores() {
    local out="" p; IFS=',' read -ra a <<< "$1"
    for p in "${a[@]}"; do
        if [[ "$p" == *-* ]]; then out="$out $(seq "${p%-*}" "${p#*-}")"; else out="$out $p"; fi
    done
    echo $out
}
ALL=($(expand_cores "$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)")); NALL=${#ALL[@]}
SANDBOX_N="${SANDBOX_N:-$NALL}"; [ "$SANDBOX_N" -gt "$NALL" ] && SANDBOX_N=$NALL
SBOX=("${ALL[@]:0:$SANDBOX_N}")
SANDBOX=$(IFS=,; echo "${SBOX[*]}")
if [ "$MODE" = "split" ]; then
    VLLM_PIN=$(IFS=,; echo "${SBOX[*]:0:$VLLM_CORES_N}")
    MAS_PIN=$(IFS=,; echo "${SBOX[*]:$VLLM_CORES_N}")
    [ -z "$MAS_PIN" ] && { echo "split 인데 MAS 코어가 0 (SANDBOX_N↑ 또는 VLLM_CORES_N↓)"; exit 1; }
else
    VLLM_PIN="$SANDBOX"; MAS_PIN="$SANDBOX"
fi

OUT="$REPO/results/fanout_starvation/${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"
echo "jobid=${SLURM_JOB_ID:-manual}  node=$(hostname)  time=$(date)"
echo "MODE=$MODE  SANDBOX_N=$SANDBOX_N  sandbox=[$SANDBOX]  vLLM=[$VLLM_PIN]  MAS=[$MAS_PIN]"
echo "backend=$FANOUT_BACKEND  corpus=$FANOUT_CORPUS  distractors=${FANOUT_DISTRACTORS:-(none=evidence-only)}"
echo "N_QUESTIONS=$N_QUESTIONS  MAX_CONCURRENCY=$MAX_CONCURRENCY  RESEARCHER_TOOL_CAP=$RESEARCHER_TOOL_CAP"

echo "vLLM 시작 (cores $VLLM_PIN)"
taskset -c "$VLLM_PIN" "$VLLM_VENV/bin/vllm" serve "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" --host 127.0.0.1 --port "$PORT" \
    --max-model-len 16384 $EAGER_FLAG \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
cleanup() {
    kill "$VLLM_PID" 2>/dev/null; pkill -P "$VLLM_PID" 2>/dev/null
}
trap cleanup EXIT
for i in $(seq 1 60); do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM 죽음 → $OUT/vllm.log"; exit 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = "200" ] && { echo "vLLM ready (~${i}0s)"; break; }
    sleep 10
done

echo "MAS 벤치 (cores $MAS_PIN, backend=$FANOUT_BACKEND)"
taskset -c "$MAS_PIN" env \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    VLLM_BASE_URL="http://127.0.0.1:$PORT/v1" VLLM_MODEL="$MODEL_NAME" \
    FANOUT_BACKEND="$FANOUT_BACKEND" \
    FANOUT_CORPUS="$FANOUT_CORPUS" \
    FANOUT_DISTRACTORS="$FANOUT_DISTRACTORS" \
    BENCH_OUT="$OUT/bench.jsonl" MA_LOG="$OUT/agents.log" LOG_FIRST_N="${LOG_FIRST_N:-0}" \
    N_QUESTIONS="$N_QUESTIONS" SEED=42 \
    MAX_CONCURRENCY="$MAX_CONCURRENCY" RESEARCHER_TOOL_CAP="$RESEARCHER_TOOL_CAP" \
    "$LG_VENV/bin/python" "$REPO/fanout_mas/run_bench_fanout.py" > "$OUT/bench_stdout.txt" 2>&1 &
BENCH_BG=$!
wait "$BENCH_BG"

echo "완료 → $OUT"
tail -n 20 "$OUT/bench_stdout.txt"
