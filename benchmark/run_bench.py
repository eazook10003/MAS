import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.topology_hotpot import build_graph, build_llm, ACTIVE_RESEARCHERS
from benchmark.tools import build_corpus_index
from benchmark.score import em, f1


HOTPOT_JSONL = Path(os.environ.get(
    "HOTPOT_JSONL",
    "/nfs/hpc/share/kangdo/personal_ma/data/hotpot/distractor_dev.jsonl",
))
BENCH_OUT = Path(os.environ.get(
    "BENCH_OUT",
    "/nfs/hpc/share/kangdo/personal_ma/logs/bench.jsonl",
))
N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "50"))
SEED = int(os.environ.get("SEED", "42"))
HOTPOT_IDS = os.environ.get("HOTPOT_IDS", "")
LOG_FIRST_N = int(os.environ.get("LOG_FIRST_N", "3"))


def load_questions(path: Path, n: int, seed: int, ids: str = "") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        all_qs = [json.loads(line) for line in f]

    if ids:
        wanted = set(ids.split(","))
        picked = [q for q in all_qs if q["id"] in wanted]
        missing = wanted - {q["id"] for q in picked}
        if missing:
            print(f"[bench] !! 못 찾은 id: {sorted(missing)}")
        return picked

    rng = random.Random(seed)
    rng.shuffle(all_qs)
    return all_qs[:n]


def passages_from_context(ctx: dict) -> list[dict]:
    return [{"title": t, "sentences": s}
            for t, s in zip(ctx["title"], ctx["sentences"])]


def summarize_timings(timings: dict) -> dict:
    # 노드별 timings -> 부하 합산(gpu=llm, cpu=tool) + critical_path(plan + max(researcher) + synth).
    gpu_total = 0.0
    cpu_total = 0.0
    cpu_by_tool = {}
    per_agent = {}
    for node_id, t in timings.items():
        llm_s = t.get("llm", 0.0)
        tool_d = t.get("tool", {})
        tool_s = sum(tool_d.values())
        gpu_total += llm_s
        cpu_total += tool_s
        for tn, ts in tool_d.items():
            cpu_by_tool[tn] = cpu_by_tool.get(tn, 0.0) + ts
        per_agent[node_id] = {
            "llm": round(llm_s, 3),
            "tool": round(tool_s, 3),
            "wall": round(t.get("wall", 0.0), 3),
        }

    plan_wall = timings.get("plan", {}).get("wall", 0.0)
    synth_wall = timings.get("synthesize", {}).get("wall", 0.0)
    researcher_walls = [
        timings[r]["wall"] for r in ACTIVE_RESEARCHERS if r in timings
    ]
    max_researcher = max(researcher_walls) if researcher_walls else 0.0
    critical_path = plan_wall + max_researcher + synth_wall

    return {
        "gpu_total": round(gpu_total, 3),
        "cpu_total": round(cpu_total, 3),
        "cpu_by_tool": {k: round(v, 3) for k, v in cpu_by_tool.items()},
        "critical_path": round(critical_path, 3),
        "per_agent": per_agent,
    }


def main():
    print(f"[bench] HOTPOT_JSONL = {HOTPOT_JSONL}")
    print(f"[bench] BENCH_OUT    = {BENCH_OUT}")
    if HOTPOT_IDS:
        print(f"[bench] HOTPOT_IDS   = {HOTPOT_IDS}  (N_QUESTIONS/SEED 무시)")
    else:
        print(f"[bench] N            = {N_QUESTIONS}  (seed={SEED})")
    print(f"[bench] LOG_FIRST_N  = {LOG_FIRST_N}  (agent I/O 로그)")

    if not HOTPOT_JSONL.exists():
        print(f"[bench] !! 데이터셋 없음: {HOTPOT_JSONL}")
        print("        submit 노드에서 benchmark/load_hotpot.py 먼저 실행")
        sys.exit(1)

    questions = load_questions(HOTPOT_JSONL, N_QUESTIONS, SEED, ids=HOTPOT_IDS)
    if not questions:
        print("[bench] !! 돌릴 문제가 없음 (id 확인)")
        sys.exit(1)
    print(f"[bench] 로드 완료: {len(questions)}문제")
    print(f"[bench] ACTIVE_RESEARCHERS = {ACTIVE_RESEARCHERS}")

    t0 = time.time()
    n_docs = build_corpus_index(HOTPOT_JSONL)
    print(f"[bench] BM25 색인 구축: {n_docs} passages ({time.time() - t0:.1f}s)")

    llm = build_llm()
    BENCH_OUT.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with open(BENCH_OUT, "w", encoding="utf-8") as fout:
        for i, q in enumerate(questions):
            qid = q["id"]
            question = q["question"]
            gold = q["answer"]
            print(f"\n[bench] {i+1}/{len(questions)}  id={qid}")
            print(f"  Q: {question}")
            print(f"  gold: {gold}")

            passages = passages_from_context(q["context"])

            t0 = time.time()
            err = None
            pred = ""
            n_tool = 0
            log_io = i < LOG_FIRST_N
            researchers = {}
            qtype_pred = ""
            timing = {}
            try:
                app = build_graph(question, passages, llm, log_io=log_io, qid=qid)
                final = app.invoke({
                    "question":   question,
                    "passages":   passages,
                    "outputs":    {},
                    "tool_calls": 0,
                    "timings":    {},
                })
                pred = final["outputs"].get("synthesize", "")
                n_tool = final.get("tool_calls", 0)
                qtype_pred = final.get("qtype", "")
                timing = summarize_timings(final.get("timings", {}))
                for r in ACTIVE_RESEARCHERS:
                    if r in final["outputs"]:
                        researchers[r] = final["outputs"][r]
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"  !! ERROR: {err}")
                traceback.print_exc()

            latency = time.time() - t0

            if pred:
                row_em = em(pred, gold)
                row_f1 = f1(pred, gold)
            else:
                row_em = 0
                row_f1 = 0.0

            row = {
                "id":          qid,
                "question":    question,
                "gold":        gold,
                "pred":        pred,
                "em":          row_em,
                "f1":          row_f1,
                "latency_s":   round(latency, 2),
                "tool_calls":  n_tool,
                "type":        q["type"],
                "qtype_pred":  qtype_pred,
                "level":       q["level"],
                "timing":      timing,
                "researchers": researchers,
                "error":       err,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(row)
            print(f"  pred: {pred!r}")
            print(f"  em={row_em}  f1={row_f1:.3f}  t={latency:.1f}s  tools={n_tool}")
            if timing:
                print(f"  e2e={latency:.1f}s  critical_path={timing['critical_path']:.1f}s"
                      f"  GPU(llm)={timing['gpu_total']:.1f}s  CPU(tool)={timing['cpu_total']:.2f}s")

    n = len(results)
    n_err = sum(1 for r in results if r["error"] is not None)
    print(f"\nAGGREGATE (n={n}, errors={n_err})")
    print(f"  EM mean      = {sum(r['em'] for r in results) / n:.3f}")
    print(f"  F1 mean      = {sum(r['f1'] for r in results) / n:.3f}")
    print(f"  latency mean = {sum(r['latency_s'] for r in results) / n:.2f}s")
    print(f"  tool calls/q = {sum(r['tool_calls'] for r in results) / n:.1f}")

    print_latency_breakdown(results)


def print_latency_breakdown(results: list[dict]):
    timed = [r for r in results if r.get("timing")]
    if not timed:
        return
    n = len(timed)

    e2e   = sum(r["latency_s"] for r in timed) / n
    crit  = sum(r["timing"]["critical_path"] for r in timed) / n
    gpu   = sum(r["timing"]["gpu_total"] for r in timed) / n
    cpu   = sum(r["timing"]["cpu_total"] for r in timed) / n

    # 툴별 CPU 합
    by_tool = {}
    for r in timed:
        for tn, ts in r["timing"]["cpu_by_tool"].items():
            by_tool[tn] = by_tool.get(tn, 0.0) + ts
    by_tool = {k: v / n for k, v in by_tool.items()}

    # agent별 GPU/CPU/wall 합
    node_order = ["plan"] + ACTIVE_RESEARCHERS + ["synthesize"]
    agg = {nid: {"llm": 0.0, "tool": 0.0, "wall": 0.0, "cnt": 0} for nid in node_order}
    for r in timed:
        for nid, pa in r["timing"]["per_agent"].items():
            if nid not in agg:
                agg[nid] = {"llm": 0.0, "tool": 0.0, "wall": 0.0, "cnt": 0}
            agg[nid]["llm"]  += pa["llm"]
            agg[nid]["tool"] += pa["tool"]
            agg[nid]["wall"] += pa["wall"]
            agg[nid]["cnt"]  += 1

    print(f"\nLATENCY BREAKDOWN (mean over {n} questions)")
    print(f"  E2E latency     = {e2e:6.2f} s")
    print(f"  critical path   = {crit:6.2f} s")
    print(f"  GPU load (LLM)  = {gpu:6.2f} s  (병렬 겹침 무시)")
    print(f"  CPU load (tool) = {cpu:6.2f} s")
    for tn, ts in sorted(by_tool.items()):
        print(f"      - {tn:<16} = {ts:6.3f} s")
    print("  per-agent (mean s):")
    print(f"    {'agent':<12}{'GPU(llm)':>10}{'CPU(tool)':>11}{'wall':>9}")
    for nid in node_order:
        a = agg.get(nid)
        if not a or a["cnt"] == 0:
            continue
        c = a["cnt"]
        print(f"    {nid:<12}{a['llm']/c:>10.2f}{a['tool']/c:>11.3f}{a['wall']/c:>9.2f}")


if __name__ == "__main__":
    main()
