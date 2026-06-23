#!/bin/bash
#SBATCH --job-name=mas_starv
#SBATCH --partition=ampere
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/kangdo/personal_ma/experiments/mas_starvation/logs/mas_starv_%j.log
#
# MAS CPU starvation 실험: GPU(vLLM)와 CPU(MAS + 무거운 툴)를 같은/다른 코어에 두고 비교.
# 코어 수 : SANDBOX_N
# 코어 배정 : MODE=shared | split(vLLM 코어 수 = VLLM_CORES_N)
#
# TOOL_CPU_HEAVY=8000 SANDBOX_N=4 MODE=split VLLM_CORES_N=2 N_QUESTIONS=100 sbatch run_mas_starvation.sh
set -u

BASE=/nfs/hpc/share/kangdo/personal_ma
HERE="$BASE/experiments/mas_starvation"
VLLM_VENV="$BASE/envs/vllm"; LG_VENV="$BASE/envs/langgraph"
MODEL_NAME="Qwen2.5-7B-Instruct"; MODEL_DIR="$BASE/models/$MODEL_NAME"
DATA="$BASE/data/hotpot/distractor_dev.jsonl"
PORT="${PORT:-$(( 8000 + (${SLURM_JOB_ID:-$RANDOM} % 1000) ))}" 

MODE="${MODE:-shared}"                        
TOOL_CPU_HEAVY="${TOOL_CPU_HEAVY:-300}"       
VLLM_CORES_N="${VLLM_CORES_N:-1}"             
N_QUESTIONS="${N_QUESTIONS:-20}"
ACTIVE_RESEARCHERS="${ACTIVE_RESEARCHERS:-searcher,reader,hop_chainer}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"           # vLLM eager (CPU에 민감하게)
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

echo "할당 ${NALL}코어  sandbox=[$SANDBOX]  MODE=$MODE  vLLM=[$VLLM_PIN]  MAS=[$MAS_PIN]"
echo "TOOL_CPU_HEAVY=$TOOL_CPU_HEAVY  N=$N_QUESTIONS  researchers=$ACTIVE_RESEARCHERS"

OUT="$HERE/results/${MODE}_n${SANDBOX_N}_h${TOOL_CPU_HEAVY}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
printf "MODE=%s SANDBOX=%s vLLM=%s MAS=%s HEAVY=%s N=%s\n" \
    "$MODE" "$SANDBOX" "$VLLM_PIN" "$MAS_PIN" "$TOOL_CPU_HEAVY" "$N_QUESTIONS" > "$OUT/meta.txt"

# vLLM 서버 (vLLM 코어에 핀)
echo "vLLM 시작 (cores $VLLM_PIN) — 모델 로딩"
taskset -c "$VLLM_PIN" "$VLLM_VENV/bin/vllm" serve "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" --host 127.0.0.1 --port "$PORT" \
    --max-model-len 16384 $EAGER_FLAG \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$OUT/vllm_server.log" 2>&1 &
VLLM_PID=$!
cleanup() { kill "$VLLM_PID" 2>/dev/null; pkill -P "$VLLM_PID" 2>/dev/null; }
trap cleanup EXIT
for i in $(seq 1 60); do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM 죽음 → $OUT/vllm_server.log"; exit 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = "200" ] && { echo "vLLM ready (~${i}0s)"; break; }
    sleep 10
done

# GPU 사용률
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 1 > "$OUT/gpu.csv" 2>&1 & GMON=$!

# vLLM 엔진 프로세스의 강제 context switch
ENGINE_PID=$(pgrep -f "EngineCore" | head -1)
if [ -n "$ENGINE_PID" ]; then
    pidstat -wt -p "$ENGINE_PID" 1 > "$OUT/ctxsw_vllm.txt" 2>&1 & CMON1=$!
else
    CMON1=""; echo "EngineCore PID 못 찾음 — vLLM context switch 측정 생략"
fi

# MAS (MAS 코어에 핀).
echo "MAS 벤치 (cores $MAS_PIN, TOOL_CPU_HEAVY=$TOOL_CPU_HEAVY)"
taskset -c "$MAS_PIN" env \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    TOOL_CPU_HEAVY="$TOOL_CPU_HEAVY" \
    VLLM_BASE_URL="http://127.0.0.1:$PORT/v1" VLLM_MODEL="$MODEL_NAME" \
    HOTPOT_JSONL="$DATA" BENCH_OUT="$OUT/bench.jsonl" \
    N_QUESTIONS="$N_QUESTIONS" SEED=42 \
    ACTIVE_RESEARCHERS="$ACTIVE_RESEARCHERS" \
    MA_LOG="$OUT/bench_agents.log" LOG_FIRST_N=0 \
    "$LG_VENV/bin/python" "$BASE/benchmark/run_bench.py" > "$OUT/bench_stdout.txt" 2>&1 &
BENCH_BG=$!
sleep 3
BENCH_PID=$(pgrep -f "run_bench.py" | head -1)
if [ -n "$BENCH_PID" ]; then
    pidstat -wt -p "$BENCH_PID" 1 > "$OUT/ctxsw_mas.txt" 2>&1 & CMON2=$!   # 강제 context switch
else
    CMON2=""
fi
wait "$BENCH_BG"

kill "$GMON" $CMON1 $CMON2 2>/dev/null
echo "완료 → $OUT"
"$LG_VENV/bin/python" "$HERE/analyze_mas.py" "$OUT"
