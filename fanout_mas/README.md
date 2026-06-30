# FanOutQA Fan-out MAS Benchmark

decompose→researchers→aggregate 멀티에이전트

## 1. 소스 코퍼스 만들기 (한 번만)

**① gold evidence (정답 페이지, ~1,593 페이지):**
```bash
FANOUT_CORPUS=$BASE/data/fanout/corpus.jsonl \
  $BASE/envs/langgraph/bin/python fanout_mas/load_fanout.py
```

**② distractors (full-wiki, ~640만 페이지):**
```bash
FANOUT_DISTRACTORS=$BASE/data/fanout/distractors.jsonl N_DISTRACTORS=6400000 HF_HUB_DISABLE_XET=1 \
  $BASE/envs/langgraph/bin/python fanout_mas/load_distractors.py
```


## 2. 백엔드 전용 파일 빌드

### 2-A. bm25s 인덱스

**full-wiki** (`fanout_index_bm25s_distract.pkl`):
```bash
sbatch fanout_mas/build_index_fanout.sh
```

**evidence-only** (`fanout_index_bm25s.pkl`):
```bash
FANOUT_DISTRACTORS= FANOUT_INDEX=$BASE/data/fanout/fanout_index_bm25s.pkl \
  sbatch fanout_mas/build_index_fanout.sh
```

### 2-B. ENNS 임베딩

**full-wiki** (`fanout_dense_emb.npy` + `fanout_dense_docs.pkl`):
```bash
sbatch fanout_mas/embed_corpus_fanout.sh
```

| 실행 백엔드 | 필요한 빌드 산출물 |
|---|---|
| bm25s evidence | `fanout_index_bm25s.pkl` |
| bm25s full-wiki | `fanout_index_bm25s_distract.pkl` |
| dense | `fanout_dense_emb.npy` + `fanout_dense_docs.pkl` |

## 3. 벤치 실행


**bm25s · evidence-only:**
```bash
FANOUT_BACKEND=bm25s FANOUT_DISTRACTORS= N_QUESTIONS=310 \
  sbatch fanout_mas/run_fanout_starvation.sh
```

**bm25s · full-wiki:**
```bash
FANOUT_BACKEND=bm25s N_QUESTIONS=310 \
  sbatch fanout_mas/run_fanout_starvation.sh
```

**ENNS · full-wiki:**
```bash
FANOUT_BACKEND=dense N_QUESTIONS=310 \
  sbatch fanout_mas/run_fanout_starvation.sh
```

### 추가 옵션

- `MODE` — `shared`(vLLM·MAS 같은 코어 공유) / `split`(분리)  default = `shared`.
- `SANDBOX_N` — 사용할 코어 수
- `VLLM_CORES_N` — `split`일 때 vLLM에 줄 코어 수 (나머지는 MAS)  default = 1
- `LOG_FIRST_N` — 앞 `N`개 문제의 에이전트 INPUT/OUTPUT 전 과정을 `agents.log`에 기록  default = 0.
