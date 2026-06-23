import json
import os
from pathlib import Path

OUT_PATH = Path(os.environ.get(
    "HOTPOT_JSONL",
    "/nfs/hpc/share/kangdo/personal_ma/data/hotpot/distractor_dev.jsonl",
))


def main():
    os.environ.setdefault("HF_HOME", "/scratch/kangdo/hf_home")
    os.environ.setdefault("TMPDIR", "/scratch/kangdo/tmp")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from datasets import load_dataset

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load_hotpot] HF_HOME = {os.environ['HF_HOME']}")
    print(f"[load_hotpot] OUT     = {OUT_PATH}")
    print(f"[load_hotpot] 다운로드 시작...")

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    print(f"[load_hotpot] 문제 수: {len(ds)}")

    n_written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in ds:
            row = {
                "id": ex["id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "type": ex["type"],
                "level": ex["level"],
                "context": ex["context"],
                "supporting_facts": ex["supporting_facts"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"[load_hotpot] 완료: {n_written}문제, {size_mb:.1f} MB → {OUT_PATH}")


if __name__ == "__main__":
    main()
