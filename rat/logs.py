"""Resumable JSONL storage.

The resumability rule of the project: every expensive result is appended to disk
the moment it exists, keyed by `qid`, and every run starts by reading which keys
are already done. A Colab disconnect at 60% then costs nothing.

A torn final line (process killed mid-write) is detected and dropped on read.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Dict, Iterable, List, Set

import pandas as pd


class JsonlStore:
    def __init__(self, path: str, key: str = "qid"):
        self.path = path
        self.key = key
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # -- reading -------------------------------------------------------
    def _iter(self) -> Iterable[dict]:
        if not os.path.exists(self.path):
            return
        bad = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
        if bad:
            warnings.warn(f"{self.path}: skipped {bad} unparseable line(s) (torn write?)")

    def load(self) -> List[dict]:
        """All records; if a key appears twice, the last one wins."""
        seen: Dict[str, dict] = {}
        for rec in self._iter():
            seen[str(rec[self.key])] = rec
        return list(seen.values())

    def load_index(self) -> Dict[str, dict]:
        return {str(r[self.key]): r for r in self.load()}

    def load_df(self) -> pd.DataFrame:
        recs = self.load()
        return pd.DataFrame(recs) if recs else pd.DataFrame(columns=[self.key])

    def done_ids(self) -> Set[str]:
        return {str(rec[self.key]) for rec in self._iter()}

    def __len__(self) -> int:
        return len(self.done_ids())

    # -- writing -------------------------------------------------------
    def append(self, rec: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def append_many(self, recs: Iterable[dict]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
            f.flush()
            os.fsync(f.fileno())


def _json_default(o):
    """Make numpy scalars / arrays serializable."""
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"not JSON serializable: {type(o)}")
