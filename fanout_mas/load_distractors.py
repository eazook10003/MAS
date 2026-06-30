import json
import os
from pathlib import Path

OUT_PATH = Path(os.environ.get(
    "FANOUT_DISTRACTORS",
    "/nfs/hpc/share/kangdo/personal_ma/data/fanout/distractors.jsonl",
))
N_DISTRACTORS = int(os.environ.get("N_DISTRACTORS", "300000"))
WIKI_CONFIG = os.environ.get("WIKI_CONFIG", "20231101.en")


def evidence_pageids() -> set:
    import fanoutqa
    ids = set()

    def walk(subs):
        for s in subs:
            ev = getattr(s, "evidence", None)
            if ev is not None:
                for e in (ev if isinstance(ev, list) else [ev]):
                    ids.add(str(e.pageid))
            walk(s.decomposition)

    for q in fanoutqa.load_dev():
        walk(q.decomposition)
    return ids


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["pageid"])
                except Exception:
                    pass
    return done


def main():
    os.environ.setdefault("HF_HOME", "/scratch/kangdo/hf_home")
    os.environ.setdefault("TMPDIR", "/scratch/kangdo/tmp")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    skip = evidence_pageids()
    done = load_done(OUT_PATH)
    print(f"[load_distractors] 목표 {N_DISTRACTORS}  (이미 {len(done)}, evidence 제외 {len(skip)})")
    print(f"[load_distractors] OUT = {OUT_PATH}")

    ds = load_dataset("wikimedia/wikipedia", WIKI_CONFIG, split="train", streaming=True)

    n_written = len(done)
    n_new = 0
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for ex in ds:
            if n_written >= N_DISTRACTORS:
                break
            pid = str(ex.get("id"))
            if pid in skip or pid in done:
                continue
            text = ex.get("text", "")
            if not text.strip():
                continue
            f.write(json.dumps({
                "pageid": pid, "title": ex.get("title", ""), "text": text,
            }, ensure_ascii=False) + "\n")
            done.add(pid)
            n_written += 1
            n_new += 1
            if n_new % 20000 == 0:
                f.flush()
                print(f"  ...{n_written}/{N_DISTRACTORS}  (이번 +{n_new})")

    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"[load_distractors] 완료: 총 {n_written} 기사, {size_mb:.0f} MB → {OUT_PATH}")


if __name__ == "__main__":
    main()
