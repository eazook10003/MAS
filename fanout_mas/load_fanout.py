import json
import os
import time
from pathlib import Path

import httpx

OUT_PATH = Path(os.environ.get(
    "FANOUT_CORPUS",
    "/nfs/hpc/share/kangdo/personal_ma/data/fanout/corpus.jsonl",
))

DELAY = float(os.environ.get("FANOUT_DELAY", "1.0"))  
MAX_RETRIES = int(os.environ.get("FANOUT_RETRIES", "6"))


def collect_evidence(questions) -> dict:
    seen = {}

    def walk(subs):
        for s in subs:
            ev = getattr(s, "evidence", None)
            if ev is not None:
                for e in (ev if isinstance(ev, list) else [ev]):
                    seen.setdefault(e.pageid, e)
            walk(s.decomposition)

    for q in questions:
        walk(q.decomposition)
    return seen


def load_done_pageids(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["pageid"])
                except Exception:
                    pass
    return done


def fetch_with_retry(ev, wiki_content, wiki_cache_dir: Path) -> str:
    cached = (wiki_cache_dir / f"{ev.pageid}-dated.md").exists()
    if not cached:
        time.sleep(DELAY)

    backoff = max(DELAY, 1.0)
    for attempt in range(MAX_RETRIES):
        try:
            return wiki_content(ev)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                ra = e.response.headers.get("retry-after")
                wait = float(ra) if (ra and ra.replace(".", "", 1).isdigit()) else backoff
                print(f"    429 rate-limited; {wait:.0f}s 대기 후 재시도 ({attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                backoff = min(backoff * 2, 60.0)
                continue
            raise
    return wiki_content(ev)


def main():
    os.environ.setdefault("TMPDIR", "/scratch/kangdo/tmp")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    import fanoutqa
    from fanoutqa import wiki_content
    from fanoutqa.wiki import WIKI_CACHE_DIR

    dev = fanoutqa.load_dev()
    pages = collect_evidence(dev)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_pageids(OUT_PATH)
    todo = [(pid, ev) for pid, ev in pages.items() if pid not in done]

    print(f"[load_fanout] dev 문제수: {len(dev)}")
    print(f"[load_fanout] 유니크 evidence 페이지: {len(pages)}  (이미 완료 {len(done)}, 남음 {len(todo)})")
    print(f"[load_fanout] DELAY={DELAY}s  OUT={OUT_PATH}")

    n_written = 0
    n_empty = 0
    n_err = 0
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for i, (pageid, ev) in enumerate(todo):
            try:
                text = fetch_with_retry(ev, wiki_content, WIKI_CACHE_DIR)
            except Exception as e:
                n_err += 1
                print(f"  !! [{i+1}/{len(todo)}] pageid={pageid} ({ev.title}) ERROR: {e}")
                continue

            if not text or not text.strip():
                n_empty += 1
                continue

            f.write(json.dumps({
                "pageid": pageid, "title": ev.title, "text": text,
            }, ensure_ascii=False) + "\n")
            f.flush()
            n_written += 1

            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(todo)}  (이번 실행 written={n_written}, err={n_err})")

    total = len(load_done_pageids(OUT_PATH))
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"[load_fanout] 완료: 이번 +{n_written} 문서 (empty={n_empty}, err={n_err})")
    print(f"[load_fanout] 코퍼스 총 {total}/{len(pages)} 문서, {size_mb:.1f} MB → {OUT_PATH}")


if __name__ == "__main__":
    main()
