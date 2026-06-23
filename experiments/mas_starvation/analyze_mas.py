import json
import re
import statistics as st
import sys
from pathlib import Path


def _med(v):
    return st.median(v) if v else None


def bench_stats(d: Path):
    f = d / "bench.jsonl"
    if not f.exists():
        return None
    e2e, tool, crit, f1, llm = [], [], [], [], []
    for line in f.read_text(errors="ignore").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error"):
            continue
        e2e.append(r.get("latency_s", 0.0))
        f1.append(r.get("f1", 0.0))
        t = r.get("timing", {})
        tool.append(t.get("cpu_total", 0.0))
        crit.append(t.get("critical_path", 0.0))
        llm.append(t.get("gpu_total", 0.0))
    return {
        "n": len(e2e),
        "e2e": _med(e2e), "tool": _med(tool), "crit": _med(crit),
        "llm": _med(llm), "f1": (sum(f1) / len(f1) if f1 else None),
    }


def gpu_stats(d: Path):
    f = d / "gpu.csv"
    if not f.exists():
        return None, None
    vals = [float(m.group(1)) for m in re.finditer(r",\s*(\d+)\s*%\s*,", f.read_text(errors="ignore"))]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    idle = 100.0 * sum(1 for v in vals if v < 20) / len(vals)
    return mean, idle


def nvcswch(d: Path, fname: str):
    # 강제 context switch(nvcswch/s) 평균.
    f = d / fname
    if not f.exists():
        return None
    text = f.read_text(errors="ignore")
    if "|__" in text:                         
        per_ts = {}                            
        for line in text.splitlines():
            p = line.split()
            if len(p) < 7 or not p[-1].startswith("|__"):
                continue                        
            try:
                per_ts[p[0]] = per_ts.get(p[0], 0.0) + float(p[-2])
            except ValueError:
                continue
        return sum(per_ts.values()) / len(per_ts) if per_ts else None
    vals = []                                   
    for line in text.splitlines():
        p = line.split()
        if len(p) < 6:
            continue
        try:
            vals.append(float(p[-2]))
        except ValueError:
            continue
    return sum(vals) / len(vals) if vals else None


def label(d: Path):
    m = d / "meta.txt"
    if m.exists():
        return m.read_text().strip().split("\n")[0]
    return d.name


def main():
    dirs = [Path(x) for x in sys.argv[1:]]
    if not dirs:
        print("usage: python analyze_mas.py <results_dir> [...]")
        raise SystemExit(1)

    def fmt(v, s="{:.2f}"):
        return s.format(v) if v is not None else "  n/a"

    print("\n" + "=" * 104)
    print("MAS CPU STARVATION — 조건 비교 (값은 중앙값)")
    print("=" * 104)
    print(f"{'condition (dir)':<32}{'e2e s':>7}{'LLM s':>7}{'tool s':>7}"
          f"{'GPU%':>6}{'idle%':>7}{'vLLMcsw':>9}{'MAScsw':>9}{'f1':>6}{'n':>4}")
    for d in dirs:
        b = bench_stats(d)
        g, idle = gpu_stats(d)
        if not b:
            print(f"{d.name:<32}  (bench.jsonl 없음)")
            continue
        vcsw = nvcswch(d, "ctxsw_vllm.txt")
        mcsw = nvcswch(d, "ctxsw_mas.txt")
        print(f"{d.name[:31]:<32}{fmt(b['e2e']):>7}{fmt(b['llm']):>7}{fmt(b['tool']):>7}"
              f"{fmt(g, '{:.0f}'):>6}{fmt(idle, '{:.0f}'):>7}"
              f"{fmt(vcsw, '{:.0f}'):>9}{fmt(mcsw, '{:.0f}'):>9}"
              f"{fmt(b['f1']):>6}{b['n']:>4}")
    print("=" * 104)


if __name__ == "__main__":
    main()
