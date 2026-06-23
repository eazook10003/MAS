# MAS latency / CPU-GPU 경합 실험

HotpotQA 멀티에이전트 시스템(MAS)에서 두 가지를 측정한다.

1. **토폴로지** — 파견하는 리서처 구성(예: searcher 포함/제외)이 latency·정확도에 주는 영향
2. **코어 배정** — vLLM(GPU)과 MAS(CPU 툴)를 같은/다른 코어에 둘 때의 CPU 경합과 GPU starvation

MAS 는 LangGraph 로 구성한다: `Planner → {searcher, reader, hop_chainer} (Send 병렬) → Synthesizer`.

## 구조

```
run_bench_lg.sh                   실험1 (topology 변경)
benchmark/                      
  run_bench.py                    
  topology_hotpot.py              LangGraph 토폴로지 (ACTIVE_RESEARCHERS 스위치)
  tools.py                        local_search(BM25), lookup, TOOL_CPU_HEAVY
  score.py                        EM/F1 채점
  load_hotpot.py                  HotpotQA 다운로드 → jsonl
experiments/mas_starvation/
  run_mas_starvation.sh           실험2 (코어 배정)
  analyze_mas.py                  결과 비교표
```

## 사전 준비 (한 번)

vLLM·langgraph venv 와 모델(`models/Qwen2.5-7B-Instruct`) 준비.

데이터 다운로드:
```bash
python benchmark/load_hotpot.py
```

## 실험 1 — 토폴로지

`ACTIVE_RESEARCHERS` 로 리서처 구성을 바꾼다.

```bash
# 리서처 3개 전부
ACTIVE_RESEARCHERS=searcher,reader,hop_chainer N_QUESTIONS=100 sbatch run_bench_lg.sh
# searcher 제외
ACTIVE_RESEARCHERS=reader,hop_chainer N_QUESTIONS=100 sbatch run_bench_lg.sh
```

출력 (`logs/`, `<jobid>` = Slurm job id):

| 파일 | 내용 |
|---|---|
| `bench_lg_<jobid>.log` | Slurm stdout — 노드, aggregate(EM/F1/latency), latency breakdown |
| `bench_<jobid>.jsonl` | 문제별 결과 (pred, em, f1, latency, 노드별 timing) |
| `vllm_<jobid>.log` | vLLM 서버 로그 |
| `bench_agents_<jobid>.log` | 앞 `LOG_FIRST_N` 문제의 agent 입출력 |
| `ctxsw_mas_<jobid>.txt` | MAS 강제 context switch (pidstat -wt) |

## 실험 2 — 코어 배정

vLLM 과 MAS 를 같은 코어(shared) 또는 분리(split)해 경합을 측정한다.

```bash
# 4코어 공유
TOOL_CPU_HEAVY=8000 SANDBOX_N=4 MODE=shared N_QUESTIONS=100 sbatch run_mas_starvation.sh
# vLLM 2 / MAS 2 분리
TOOL_CPU_HEAVY=8000 SANDBOX_N=4 MODE=split VLLM_CORES_N=2 N_QUESTIONS=100 sbatch run_mas_starvation.sh
```

| 노브 | 뜻 | 기본 |
|---|---|---|
| `MODE` | `shared`(공유) / `split`(분리) | shared |
| `SANDBOX_N` | 쓸 코어 수 (엣지 흉내) | 할당 전체 |
| `VLLM_CORES_N` | split 일 때 vLLM 코어 수 (나머지 = MAS) | 1 |
| `TOOL_CPU_HEAVY` | 검색당 합성 CPU 부하 (0 = 원본 BM25) | 300 |
| `N_QUESTIONS` | 문제 수 | 20 |


출력 (run 마다 `experiments/mas_starvation/results/<MODE>_n<SANDBOX_N>_h<HEAVY>_<시각>/`):

| 파일 | 내용 |
|---|---|
| `meta.txt` | 이 run 설정 (MODE, vLLM/MAS 코어, HEAVY, N) |
| `bench.jsonl` | 문제별 결과 + timing |
| `bench_stdout.txt` | 벤치 stdout |
| `gpu.csv` | GPU 사용률 타임라인 (1초 간격) |
| `ctxsw_vllm.txt` | vLLM 강제 context switch (pidstat -wt) |
| `ctxsw_mas.txt` | MAS 강제 context switch (pidstat -wt) |
| `vllm_server.log` | vLLM 서버 로그 |

Slurm stdout 은 `experiments/mas_starvation/logs/mas_starv_<jobid>.log` 에 남고, run 끝에 `analyze_mas.py` 가 자동으로 비교표를 출력한다. 여러 run 을 직접 비교하려면:

```bash
python experiments/mas_starvation/analyze_mas.py results/shared_* results/split_*
```

비교표 컬럼: `e2e / LLM / tool / GPU% / idle% / vLLMcsw / MAScsw / f1`.
- **idle%** = GPU util < 20 인 샘플 비율 (GPU 가 논 시간)
- **csw** = 강제 context switch/s (경합 지표; MAS·vLLM 각각)
