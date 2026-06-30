import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
from langchain_core.tools import tool

from fanoutqa.retrieval import chunk_text, RetrievalResult

BACKEND = os.environ.get("FANOUT_BACKEND", "rank_bm25")

CORPUS_JSONL = os.environ.get(
    "FANOUT_CORPUS",
    "/nfs/hpc/share/kangdo/personal_ma/data/fanout/corpus.jsonl",
)
DISTRACTORS_JSONL = os.environ.get("FANOUT_DISTRACTORS", "")
DOC_LEN = int(os.environ.get("FANOUT_DOC_LEN", "2048"))


_DOCS = None     


def _has_distractors() -> bool:
    return bool(DISTRACTORS_JSONL) and Path(DISTRACTORS_JSONL).exists()


def _source_paths(primary: str) -> list[str]:
    paths = [primary]
    if _has_distractors():
        paths.append(DISTRACTORS_JSONL)
    return paths


def _iter_chunks(paths: list[str]):
    docs = []
    chunk_texts = []
    n_pages = 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                n_pages += 1
                for chunk in chunk_text(d["text"], max_chunk_size=DOC_LEN):
                    docs.append(RetrievalResult(d["title"], chunk))
                    chunk_texts.append(chunk)
    return docs, chunk_texts, n_pages


# =====================================================================
# bm25 백엔드 — (순수 파이썬, GIL 묶임)
# =====================================================================
import ftfy
from rank_bm25 import BM25Plus
import fanoutqa.norm as fnorm                 
from fanoutqa.norm import normalize

_INDEX = None
_NLP_FAST = None
NPROC = int(os.environ.get("FANOUT_NPROC", "8"))


def _get_fast_nlp():
    global _NLP_FAST
    if _NLP_FAST is None:
        import spacy
        _NLP_FAST = spacy.load("en_core_web_sm")
    return _NLP_FAST


def _tokenize(text: str) -> list[str]:
    return normalize(text).split(" ")


def _tokenize_many(texts: list[str], batch_size: int = 256) -> list[list[str]]:
    nlp = _get_fast_nlp()
    pres = [fnorm.normalize_numbers(ftfy.fix_text(str(t).lower())) for t in texts]
    out = []
    for doc in nlp.pipe(pres, batch_size=batch_size, n_process=NPROC):
        lem = " ".join(tok.lemma_ for tok in doc)
        post = fnorm.normalize_whitespace(fnorm.remove_punct(lem))
        out.append(post.split(" "))
    return out


def _build_rank_bm25(paths: list[str]):
    docs, chunk_texts, n_pages = _iter_chunks(paths)
    tokenized = _tokenize_many(chunk_texts)
    index = BM25Plus(tokenized)
    return index, docs, n_pages


def _search_rank_bm25(query: str, top_k: int) -> list[RetrievalResult]:
    scores = _INDEX.get_scores(_tokenize(query)) 
    idxs = np.argsort(scores)[::-1][:top_k]
    return [_DOCS[i] for i in idxs]


# =====================================================================
# bm25s 백엔드 —  (numpy/scipy, GIL 풂)
# =====================================================================
def _build_bm25s(paths: list[str]):
    import bm25s
    docs, chunk_texts, n_pages = _iter_chunks(paths)
    ctok = bm25s.tokenize(chunk_texts, stopwords="en", show_progress=False)
    index = bm25s.BM25()
    index.index(ctok, show_progress=False)
    return index, docs, n_pages


def _search_bm25s(query: str, top_k: int) -> list[RetrievalResult]:
    import bm25s
    qtok = bm25s.tokenize(query, stopwords="en", show_progress=False)
    res, _ = _INDEX.retrieve(qtok, k=top_k, n_threads=1, show_progress=False)
    return [_DOCS[i] for i in res[0]]


# =====================================================================
# dense 백엔드 — 임베딩 ENNS (matmul, GIL 풂)
# =====================================================================
EMB_PATH = os.environ.get("FANOUT_EMB",
    "/nfs/hpc/share/kangdo/personal_ma/data/fanout/fanout_dense_emb.npy")
EMB_DOCS_PATH = os.environ.get("FANOUT_EMB_DOCS",
    "/nfs/hpc/share/kangdo/personal_ma/data/fanout/fanout_dense_docs.pkl")
EMB_MODEL = os.environ.get("FANOUT_EMB_MODEL",
    "/nfs/hpc/share/kangdo/personal_ma/data/fanout/emb_model")  
_EMB = None   
_MODEL = None 


def _get_emb_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMB_MODEL, device="cpu") 
    return _MODEL


def _load_dense() -> int:
    global _EMB, _DOCS
    _EMB = np.load(EMB_PATH) 
    with open(EMB_DOCS_PATH, "rb") as f:
        _DOCS = pickle.load(f)
    _get_emb_model() 
    return len(_DOCS)


def _search_dense(query: str, top_k: int) -> list[RetrievalResult]:
    q = _get_emb_model().encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores = (q @ _EMB.T)[0]                
    idx = np.argpartition(scores, -top_k)[-top_k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [_DOCS[i] for i in idx]


# =====================================================================
# 백엔드 선택
# =====================================================================
def _default_pickle() -> str:
    env = os.environ.get("FANOUT_INDEX", "")
    if env:
        return env
    base = "/nfs/hpc/share/kangdo/personal_ma/data/fanout"
    eng = "_bm25s" if BACKEND == "bm25s" else ""
    tag = "_distract" if _has_distractors() else ""
    return f"{base}/fanout_index{eng}{tag}.pkl"


def build_corpus_index(jsonl_path: str = None, pickle_path: str = None,
                       force: bool = False) -> int:
    global _INDEX, _DOCS
    if BACKEND == "dense":
        return _load_dense()

    jsonl_path = jsonl_path or CORPUS_JSONL
    pickle_path = pickle_path or _default_pickle()
    pk = Path(pickle_path)
    paths = _source_paths(jsonl_path)

    src_mtime = max(Path(p).stat().st_mtime for p in paths)
    fresh = not force and pk.exists() and pk.stat().st_mtime >= src_mtime
    if fresh:
        with open(pk, "rb") as f:
            _INDEX, _DOCS = pickle.load(f)
        return len(_DOCS)

    t0 = time.time()
    if BACKEND == "bm25s":
        _INDEX, _DOCS, n_pages = _build_bm25s(paths)
    else:
        _INDEX, _DOCS, n_pages = _build_rank_bm25(paths)
    pk.parent.mkdir(parents=True, exist_ok=True)
    with open(pk, "wb") as f:
        pickle.dump((_INDEX, _DOCS), f)
    print(f"[tools_fanout] ({BACKEND}) 인덱스 빌드: {n_pages} 페이지 → {len(_DOCS)} 청크 "
          f"({time.time()-t0:.1f}s) → {pk}")
    return len(_DOCS)


def search_raw(query: str, top_k: int = 3) -> list[RetrievalResult]:
    if BACKEND == "dense":
        if _EMB is None:
            raise RuntimeError("dense embeddings not loaded — call build_corpus_index first")
        return _search_dense(query, top_k)
    if _INDEX is None:
        raise RuntimeError("index not built — call build_corpus_index first")
    if BACKEND == "bm25s":
        return _search_bm25s(query, top_k)
    return _search_rank_bm25(query, top_k)


def make_local_search():
    @tool
    def local_search(query: str, top_k: int = 3) -> str:
        """Search the local Wikipedia corpus and return the top_k most relevant passages (title + fragment)."""
        hits = search_raw(query, top_k)
        return "\n\n".join(f"[{h.title}]\n{h.content}" for h in hits)

    return local_search
