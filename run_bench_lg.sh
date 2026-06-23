#!/bin/bash
#SBATCH --job-name=bench_lg
#SBATCH --partition=ampere
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/kangdo/personal_ma/logs/bench_lg_%j.log
#
# 토폴로지 실험: vLLM 서버를 띄우고 HotpotQA MAS 벤치를 돌린다.
# ACTIVE_RESEARCHERS=reader,hop_chainer N_QUESTIONS=100 sbatch run_bench_lg.sh

BASE="/nfs/hpc/share/kangdo/personal_ma"
MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_DIR="$BASE/models/$MODEL_NAME"
PORT=$(( 8000 + (${SLURM_JOB_ID:-$RANDOM} % 1000) ))

VLLM_VENV="$BASE/envs/vllm"
LG_VENV="$BASE/envs/langgraph"
VLLM_LOG="$BASE/logs/vllm_${SLURM_JOB_ID}.log"


setup_env() {
    export CUDA_HOME="/usr/local/apps/cuda/13.0"
    export CPATH="/usr/local/apps/python/3.12/include/python3.12"
    export TMPDIR="/scratch/kangdo/tmp"
    export HF_HOME="/scratch/kangdo/hf_home"
    export VLLM_CACHE_ROOT="/scratch/kangdo/vllm_cache"
    export TORCHINDUCTOR_CACHE_DIR="/scratch/kangdo/inductor_cache"
    export TRITON_CACHE_DIR="/scratch/kangdo/triton_cache"
    mkdir -p "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT" \
             "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
}


start_vllm() {
    echo "vLLM 서버 시작"
    source "$VLLM_VENV/bin/activate"
    vllm serve "$MODEL_DIR" \
        --served-model-name "$MODEL_NAME" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --max-model-len 16384 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID=$VLLM_PID  log=$VLLM_LOG"
    deactivate
}


wait_for_vllm() {
    echo "vLLM /health 대기 (최대 10분)"
    local i code
    for i in $(seq 1 60); do
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM 죽음 → $VLLM_LOG"; return 1; }
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health")
        [ "$code" = "200" ] && { echo "vLLM ready (~${i}0s)"; return 0; }
        echo "  ...$i/60  http=$code"
        sleep 10
    done
    echo "10분 안에 vLLM 이 안 뜸. 중단."
    return 1
}


run_bench() {
    echo "benchmark 실행"
    source "$LG_VENV/bin/activate"

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
    export VLLM_MODEL="$MODEL_NAME"
    export HOTPOT_JSONL="$BASE/data/hotpot/distractor_dev.jsonl"
    export BENCH_OUT="$BASE/logs/bench_${SLURM_JOB_ID}.jsonl"
    export N_QUESTIONS="${N_QUESTIONS:-50}"
    export SEED="${SEED:-42}"
    export HOTPOT_IDS="${HOTPOT_IDS:-}"                   # 지정 시 N_QUESTIONS/SEED 무시
    export ACTIVE_RESEARCHERS="${ACTIVE_RESEARCHERS:-searcher,reader,hop_chainer}"
    export MA_LOG="$BASE/logs/bench_agents_${SLURM_JOB_ID}.log"
    export LOG_FIRST_N="${LOG_FIRST_N:-3}"               # 앞 N문제만 agent I/O 로그

    # MAS 를 백그라운드로 띄우고 그 PID 의 강제 context switch(nvcswch) 를 pidstat 로 1초마다 측정.
    local CSW_OUT="$BASE/logs/ctxsw_mas_${SLURM_JOB_ID}.txt"
    python "$BASE/benchmark/run_bench.py" &
    local BENCH_PID=$!
    echo "MAS PID=$BENCH_PID  ctxsw=$CSW_OUT"
    pidstat -wt -p "$BENCH_PID" 1 > "$CSW_OUT" 2>&1 & local PIDSTAT_PID=$!
    wait "$BENCH_PID"; local rc=$?
    kill "$PIDSTAT_PID" 2>/dev/null
    return $rc
}


cleanup() {
    if [ -n "${VLLM_PID:-}" ]; then
        echo "cleanup: vLLM($VLLM_PID) 종료"
        kill "$VLLM_PID" 2>/dev/null
        wait "$VLLM_PID" 2>/dev/null
    fi
}


main() {
    echo "node=$(hostname)  time=$(date)"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    setup_env
    start_vllm
    trap cleanup EXIT
    wait_for_vllm || exit 1
    run_bench
    local rc=$?
    echo "done rc=$rc  time=$(date)"
    exit $rc
}


main "$@"
