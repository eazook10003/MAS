import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from fanoutqa.retrieval import chunk_text, RetrievalResult

CORPUS = os.environ.get("FANOUT_CORPUS",
    "/nfs/hpc/share/kangdo/personal_ma/data/fanout/corpus.jsonl")
DISTRACTORS = os.environ.get("FANOUT_DISTRACTORS",
    "/nfs/hpc/share/kangdo/personal_ma/data/fanout/distractors.jsonl")
DOC_LEN = int(os.environ.get("FANOUT_DOC_LEN", "2048"))
MODEL = os.environ.get("FANOUT_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH = int(os.environ.get("EMB_BATCH", "512"))

OUT_EMB = os.environ.get("FANOUT_EMB",
    "/nfs/hpc/share/kangdo/personal_ma/data/fanout/fanout_dense_emb.npy")
OUT_DOCS = os.environ.get("FANOUT_EMB_DOCS",
    "/nfs/hpc/share/kangdo/personal_ma/data/fanout/fanout_dense_docs.pkl")


def main():
    paths = [CORPUS] + ([DISTRACTORS] if DISTRACTORS and Path(DISTRACTORS).exists() else [])
    docs = []
    texts = []
    n_pages = 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                n_pages += 1
                for chunk in chunk_text(d["text"], max_chunk_size=DOC_LEN):
                    docs.append(RetrievalResult(d["title"], chunk))
                    texts.append(chunk)
    print(f"[embed] {n_pages} 페이지 → {len(texts)} 청크", flush=True)

    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embed] 모델 {MODEL}  device={dev}  batch={BATCH}", flush=True)
    model = SentenceTransformer(MODEL, device=dev)

    t0 = time.time()
    emb = model.encode(
        texts, batch_size=BATCH, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,   
    ).astype("float32")
    print(f"[embed] 인코딩 {emb.shape} ({time.time()-t0:.0f}s)", flush=True)

    Path(OUT_EMB).parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMB, emb)
    with open(OUT_DOCS, "wb") as f:
        pickle.dump(docs, f)
    print(f"[embed] 저장: {OUT_EMB} ({emb.nbytes/1e9:.1f}GB), {OUT_DOCS}", flush=True)


if __name__ == "__main__":
    main()
