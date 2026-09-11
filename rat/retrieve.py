"""Retrieval tier 1: BM25 over Wikipedia (DPR 100-word passages) via Pyserini.

Why Pyserini: it ships a prebuilt Lucene index (`wikipedia-dpr-100w`, ~2.5 GB)
so no corpus indexing happens on Colab. The index is downloaded once into a
local cache, then copied to Drive; later sessions copy it back (Lucene random
access directly on the Drive FUSE mount is too slow to search from).

Pyserini needs Java >= 21. See JAVA_HELP for the Colab install lines.

A `CachedJsonlRetriever` is provided as a fallback: point `cached_retrieval_file`
at a JSONL with {"qid" | "question", "ctxs": [{"title","text","score"?}]} lines
(e.g. retrievals released by prior work) and set retriever: cached_jsonl.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from .logs import JsonlStore

JAVA_HELP = (
    "Pyserini needs Java >= 21. On Colab run, in a cell:\n"
    "  !apt-get install -y -qq openjdk-21-jdk-headless\n"
    "then in Python:\n"
    "  import os, glob; jh = sorted(glob.glob('/usr/lib/jvm/java-21-openjdk*'))[-1]\n"
    "  os.environ['JAVA_HOME'] = jh; os.environ['PATH'] = jh + '/bin:' + os.environ['PATH']"
)


def check_java(min_major: int = 21) -> int:
    exe = shutil.which("java")
    if exe is None:
        raise RuntimeError("java not found on PATH.\n" + JAVA_HELP)
    proc = subprocess.run(["java", "-version"], capture_output=True, text=True)
    out = (proc.stderr or "") + (proc.stdout or "")
    m = re.search(r'version "(\d+)(?:\.(\d+))?', out)
    if not m:
        raise RuntimeError(f"Could not parse java version from: {out!r}\n" + JAVA_HELP)
    major = int(m.group(1))
    if major == 1 and m.group(2):  # "1.8.0" style
        major = int(m.group(2))
    if major < min_major:
        raise RuntimeError(f"Java {major} found, need >= {min_major}.\n" + JAVA_HELP)
    return major


def stage_pyserini_cache(local: str, drive: Optional[str]) -> None:
    """Copy a persisted index from Drive to local disk (if present) and point
    Pyserini's cache at the local directory."""
    os.makedirs(local, exist_ok=True)
    if drive:
        src, dst = os.path.join(drive, "indexes"), os.path.join(local, "indexes")
        if os.path.isdir(src) and not os.path.isdir(dst):
            print(f"[retrieve] copying persisted index Drive -> local ({src})")
            shutil.copytree(src, dst)
    os.environ["PYSERINI_CACHE"] = local


def persist_pyserini_cache(local: str, drive: Optional[str]) -> None:
    if not drive:
        return
    src, dst = os.path.join(local, "indexes"), os.path.join(drive, "indexes")
    if os.path.isdir(src) and not os.path.isdir(dst):
        print(f"[retrieve] persisting index local -> Drive ({dst}); one-time copy")
        shutil.copytree(src, dst)


def _parse_raw(raw: str) -> Dict[str, str]:
    """DPR-format passages are stored as {"id":..., "contents": "\"Title\"\\nText"}."""
    contents = raw
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            contents = obj.get("contents", obj.get("text", raw))
    except (json.JSONDecodeError, TypeError):
        pass
    if "\n" in contents:
        title, text = contents.split("\n", 1)
    else:
        title, text = "", contents
    return {"title": title.strip().strip('"'), "text": " ".join(text.split())}


class BM25Retriever:
    name = "bm25"

    def __init__(self, index_name: str, cache_local: str, cache_drive: Optional[str] = None, threads: int = 8):
        check_java()
        stage_pyserini_cache(cache_local, cache_drive)
        from pyserini.search.lucene import LuceneSearcher  # noqa: WPS433 (heavy import kept local)

        try:
            self.searcher = LuceneSearcher.from_prebuilt_index(index_name)
        except Exception as e:  # noqa: BLE001
            try:
                names = list(LuceneSearcher.list_prebuilt_indexes())  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                names = []
            hint = [n for n in names if "wiki" in n.lower()]
            raise RuntimeError(f"Could not load prebuilt index {index_name!r}: {e}\nWikipedia-ish prebuilt names: {hint}") from e
        persist_pyserini_cache(cache_local, cache_drive)
        self.threads = threads
        self.index_name = index_name

    def _doc(self, docid: str) -> Dict[str, str]:
        d = self.searcher.doc(docid)
        if d is None:
            return {"title": "", "text": ""}
        return _parse_raw(d.raw())

    def search_batch(self, qids: List[str], queries: List[str], k: int) -> Dict[str, List[dict]]:
        hits = self.searcher.batch_search(queries, qids, k=k, threads=self.threads)
        out: Dict[str, List[dict]] = {}
        for qid in qids:
            res = []
            for rank, h in enumerate(hits.get(qid, [])):
                doc = self._doc(h.docid)
                res.append({"rank": rank + 1, "docid": str(h.docid), "score": float(h.score), **doc})
            out[qid] = res
        return out


class CachedJsonlRetriever:
    """Serve retrievals from a pre-computed JSONL (no index needed)."""

    name = "cached_jsonl"

    def __init__(self, path: str):
        self.by_qid: Dict[str, List[dict]] = {}
        self.by_question: Dict[str, List[dict]] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ctxs = rec.get("ctxs") or rec.get("hits") or []
                hits = [
                    {"rank": i + 1, "docid": str(c.get("id", c.get("docid", i))), "score": float(c.get("score", 0.0)),
                     "title": str(c.get("title", "")), "text": str(c.get("text", ""))}
                    for i, c in enumerate(ctxs)
                ]
                if "qid" in rec:
                    self.by_qid[str(rec["qid"])] = hits
                if "id" in rec:
                    self.by_qid[str(rec["id"])] = hits
                if "question" in rec:
                    self.by_question[str(rec["question"]).strip()] = hits

    def search_batch(self, qids: List[str], queries: List[str], k: int) -> Dict[str, List[dict]]:
        out = {}
        for qid, q in zip(qids, queries):
            hits = self.by_qid.get(qid) or self.by_question.get(q.strip()) or []
            out[qid] = hits[:k]
        return out


def get_retriever(cfg):
    from .config import paths

    if cfg.retriever == "bm25":
        return BM25Retriever(cfg.index_name, cfg.pyserini_cache_local, paths(cfg)["pyserini_cache"], cfg.search_threads)
    if cfg.retriever == "cached_jsonl":
        if not cfg.cached_retrieval_file:
            raise ValueError("retriever=cached_jsonl requires cached_retrieval_file")
        return CachedJsonlRetriever(cfg.cached_retrieval_file)
    raise ValueError(f"Unknown retriever {cfg.retriever!r} (Phase 1 supports: bm25, cached_jsonl)")


def retrieve_all(retriever, df: pd.DataFrame, k: int, store: JsonlStore, batch: int = 64) -> None:
    """Resumable: only queries whose qid is not yet in `store` are searched."""
    done = store.done_ids()
    todo = df[~df["qid"].astype(str).isin(done)].reset_index(drop=True)
    print(f"[retrieve] {len(done)} cached, {len(todo)} to search (k={k})")
    for start in tqdm(range(0, len(todo), batch), desc="retrieve", disable=len(todo) == 0):
        chunk = todo.iloc[start:start + batch]
        qids = chunk["qid"].astype(str).tolist()
        res = retriever.search_batch(qids, chunk["question"].astype(str).tolist(), k)
        store.append_many(
            {"qid": qid, "question": q, "retriever": getattr(retriever, "name", "?"), "hits": res.get(qid, [])}
            for qid, q in zip(qids, chunk["question"].tolist())
        )
