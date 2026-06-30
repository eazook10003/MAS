import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fanout_mas.topology_fanout import build_graph, build_llm
from fanout_mas.tools_fanout import build_corpus_index
from fanout_mas.score_fanout import Scorer, ROUGE_TYPES

import fanoutqa


BENCH_OUT = Path(os.environ.get(
    "BENCH_OUT",
    "/nfs/hpc/share/kangdo/personal_ma/logs/bench_fanout.jsonl",
))
N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "20"))
SEED = int(os.environ.get("SEED", "42"))
FANOUT_IDS = os.environ.get("FANOUT_IDS", "")
LOG_FIRST_N = int(os.environ.get("LOG_FIRST_N", "2"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "0"))


def load_questions(n: int, seed: int, ids: str = ""):
    dev = fanoutqa.load_dev()
    if ids:
        wanted = set(ids.split(","))
        return [q for q in dev if q.id in wanted]
    rng = random.Random(seed)
    idx = list(range(len(dev)))
    rng.shuffle(idx)
    return [dev[i] for i in idx[:n]]


def summarize_timings(timings: dict) -> dict:
    gpu_total = 0.0
    cpu_total = 0.0
    cpu_by_tool = {}
    per_agent = {}
    researcher_walls = []
    for nid, t in timings.items():
        llm_s = t.get("llm", 0.0)
        tool_d = t.get("tool", {})
        tool_s = sum(tool_d.values())
        gpu_total += llm_s
        cpu_total += tool_s
        for tn, ts in tool_d.items():
            cpu_by_tool[tn] = cpu_by_tool.get(tn, 0.0) + ts
        per_agent[nid] = {"llm": round(llm_s, 3), "tool": round(tool_s, 3),
                          "wall": round(t.get("wall", 0.0), 3)}
        if nid.startswith("researcher_"):
            researcher_walls.append(t.get("wall", 0.0))

    dec = timings.get("decompose", {}).get("wall", 0.0)
    agg = timings.get("aggregate", {}).get("wall", 0.0)
    max_r = max(researcher_walls) if researcher_walls else 0.0
    critical = dec + max_r + agg          # decompose + 가장 느린 researcher + aggregate

    return {
        "gpu_total": round(gpu_total, 3),
        "cpu_total": round(cpu_total, 3),
        "cpu_by_tool": {k: round(v, 3) for k, v in cpu_by_tool.items()},
        "critical_path": round(critical, 3),
        "fanout": len(researcher_walls),
        "per_agent": per_agent,
    }


def main():
    print(f"[bench_fanout] BENCH_OUT = {BENCH_OUT}")
    if FANOUT_IDS:
        print(f"[bench_fanout] FANOUT_IDS = {FANOUT_IDS}")
    else:
        print(f"[bench_fanout] N = {N_QUESTIONS}  (seed={SEED})")
    mc_str = "unlimited" if MAX_CONCURRENCY <= 0 else str(MAX_CONCURRENCY)
    print(f"[bench_fanout] MAX_CONCURRENCY = {mc_str}  LOG_FIRST_N = {LOG_FIRST_N}")

    questions = load_questions(N_QUESTIONS, SEED, FANOUT_IDS)
    if not questions:
        print("[bench_fanout] !! 돌릴 문제가 없음 (id 확인)")
        sys.exit(1)
    print(f"[bench_fanout] 로드: {len(questions)}문제")

    t0 = time.time()
    n_chunks = build_corpus_index()
    print(f"[bench_fanout] 색인 로드/빌드: {n_chunks} 청크 ({time.time()-t0:.1f}s)")

    llm = build_llm()

    rows = []
    answers = []                        
    for i, q in enumerate(questions):
        print(f"\n[bench_fanout] {i+1}/{len(questions)}  id={q.id}")
        print(f"  Q: {q.question}")

        t_start = time.time()
        err = None
        pred = ""
        n_tool = 0
        n_sub = 0
        timing = {}
        log_io = i < LOG_FIRST_N
        try:
            app = build_graph(llm, log_io=log_io, qid=q.id)
            cfg = {"max_concurrency": MAX_CONCURRENCY} if MAX_CONCURRENCY > 0 else {}
            final = app.invoke(
                {"question": q.question, "sub_questions": [],
                 "researcher_outputs": [], "outputs": {},
                 "tool_calls": 0, "timings": {}},
                config=cfg,
            )
            pred = final["outputs"].get("aggregate", "")
            n_tool = final.get("tool_calls", 0)
            n_sub = len(final.get("sub_questions", []))
            timing = summarize_timings(final.get("timings", {}))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"  !! ERROR: {err}")
            traceback.print_exc()

        latency = time.time() - t_start
        answers.append({"id": q.id, "answer": pred})
        rows.append({
            "id": q.id, "question": q.question, "gold": q.answer,
            "pred": pred, "latency_s": round(latency, 2), "tool_calls": n_tool,
            "n_subq": n_sub, "categories": q.categories,
            "timing": timing, "error": err,
        })
        print(f"  pred: {pred[:120]!r}")
        print(f"  t={latency:.1f}s  subq={n_sub} tools={n_tool}")
        if timing:
            print(f"  critical_path={timing['critical_path']:.1f}s  "
                  f"GPU(llm)={timing['gpu_total']:.1f}s  CPU(tool)={timing['cpu_total']:.2f}s  "
                  f"fanout={timing['fanout']}")


    scorer = Scorer(questions, answers)
    acc, acc_raw = scorer.score_accuracy()          
    rouge, rouge_raw = scorer.score_rouge()        

    # 문제별 점수 붙여 jsonl 작성
    BENCH_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCH_OUT, "w", encoding="utf-8") as fout:
        for row in rows:
            rg = rouge_raw[row["id"]]
            row["loose_acc"] = round(acc_raw[row["id"]], 3)
            row["rouge"] = {k: {"p": round(getattr(rg, k).precision, 3),
                                "r": round(getattr(rg, k).recall, 3),
                                "f": round(getattr(rg, k).fscore, 3)}
                            for k in ROUGE_TYPES}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(rows)
    n_err = sum(1 for r in rows if r["error"])
    print(f"\nAGGREGATE (n={n}, errors={n_err})")
    print(f"  loose acc    = {acc.loose:.3f}")
    print(f"  strict acc   = {acc.strict:.3f}")
    print("  ROUGE (P / R / F1):")
    for k in ROUGE_TYPES:
        part = getattr(rouge, k)
        print(f"    {k:8} = {part.precision:.3f} / {part.recall:.3f} / {part.fscore:.3f}")
    print(f"  latency mean = {sum(r['latency_s'] for r in rows)/n:.2f}s")
    print(f"  subq mean    = {sum(r['n_subq'] for r in rows)/n:.1f}")
    print(f"  tools/q mean = {sum(r['tool_calls'] for r in rows)/n:.1f}")

    timed = [r for r in rows if r.get("timing")]
    if timed:
        m = len(timed)
        print(f"\nLATENCY BREAKDOWN (mean over {m})")
        print(f"  critical path = {sum(r['timing']['critical_path'] for r in timed)/m:6.2f} s")
        print(f"  GPU load(LLM) = {sum(r['timing']['gpu_total'] for r in timed)/m:6.2f} s")
        print(f"  CPU load(tool)= {sum(r['timing']['cpu_total'] for r in timed)/m:6.2f} s")
        print(f"  fanout mean   = {sum(r['timing']['fanout'] for r in timed)/m:6.1f}")

        agg = {k: {"llm": 0.0, "tool": 0.0, "wall": 0.0, "cnt": 0}
               for k in ("decompose", "researcher", "aggregate")}
        for r in timed:
            for nid, pa in r["timing"]["per_agent"].items():
                key = "researcher" if nid.startswith("researcher_") else nid
                if key in agg:
                    agg[key]["llm"] += pa["llm"]
                    agg[key]["tool"] += pa["tool"]
                    agg[key]["wall"] += pa["wall"]
                    agg[key]["cnt"] += 1
        print(f"  per-agent (mean s):")
        print(f"    {'agent':<12}{'GPU(llm)':>10}{'CPU(tool)':>11}{'wall':>9}{'cnt':>7}")
        for k in ("decompose", "researcher", "aggregate"):
            a = agg[k]
            if a["cnt"]:
                c = a["cnt"]
                print(f"    {k:<12}{a['llm']/c:>10.2f}{a['tool']/c:>11.3f}{a['wall']/c:>9.2f}{a['cnt']:>7}")


if __name__ == "__main__":
    main()
