import json
import os

import numpy as np
from langchain_core.tools import tool
from rank_bm25 import BM25Okapi

_BM25 = None
_CORPUS = None


def _tokenize(s: str) -> list[str]:
    return s.lower().split()


def _cpu_heavy_work(factor: int, _dim: int = 512) -> float:
    # matmul
    if factor <= 0:
        return 0.0
    rng = np.random.default_rng(0)
    a = rng.random((_dim, _dim), dtype=np.float32)
    b = rng.random((_dim, _dim), dtype=np.float32)
    s = 0.0
    for _ in range(factor):
        c = a @ b
        s += float(c.mean())
        a = c / (c.max() + 1e-6)
    return s


def build_corpus_index(jsonl_path) -> int:
    global _BM25, _CORPUS

    seen = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            ctx = q["context"]
            for title, sents in zip(ctx["title"], ctx["sentences"]):
                if title not in seen:
                    seen[title] = " ".join(sents)

    _CORPUS = []
    tokenized = []
    for title, text in seen.items():
        _CORPUS.append({"title": title, "text": text})
        tokenized.append(_tokenize(title + " " + text))

    _BM25 = BM25Okapi(tokenized)
    return len(_CORPUS)


def make_local_search():
    heavy = int(os.environ.get("TOOL_CPU_HEAVY", "0"))

    @tool
    def local_search(query: str, top_k: int = 3) -> str:
        """Search the whole HotpotQA passage corpus with BM25 and return the top_k most relevant passages."""
        if _BM25 is None:
            return "ERROR: BM25 index not built"
        scores = _BM25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        parts = []
        for i in order[:top_k]:
            doc = _CORPUS[i]
            parts.append(f"[{doc['title']}]\n{doc['text']}")

        if heavy > 0:
            _cpu_heavy_work(heavy)

        return "\n\n".join(parts)

    return local_search


def make_lookup(passages: list[dict]):
    @tool
    def lookup(keyword: str) -> str:
        """Find sentences containing the keyword within this question's 10 reference passages. Returns up to 5 matching sentences with their passage titles."""
        kw = keyword.lower()
        hits = []
        for p in passages:
            for sent in p["sentences"]:
                if kw in sent.lower():
                    hits.append(f"[{p['title']}] {sent.strip()}")
                    if len(hits) >= 5:
                        return "\n".join(hits)
        if hits:
            return "\n".join(hits)
        return f"(no sentence contains '{keyword}')"

    return lookup
