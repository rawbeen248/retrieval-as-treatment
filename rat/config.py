"""Configuration and workdir layout.

Everything generated lives under `cfg.workdir` (a Google Drive path on Colab):

    workdir/
      data/            unified query tables (parquet) + sampling metadata
      retrievals/      cached top-k passages per (dataset, retriever)   [resumable JSONL]
      logs/gen/        generations per (dataset, model, retriever, arm) [resumable JSONL]
      features/        F0/F1 feature tables joined with outcomes (parquet)
      results/phase1/  summary.json, gate_curve.csv, report.md
      figures/phase1/  png figures
      pyserini_cache/  persisted copy of the BM25 index (so it downloads once)
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import yaml


def safe(s: str) -> str:
    """Filesystem-safe tag for model / retriever names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", s)


@dataclass
class Config:
    workdir: str
    dataset: str = "popqa"
    n_queries: int = 2000
    seed: int = 42
    pop_bins: int = 5

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    model_revision: Optional[str] = None
    load_in_4bit: bool = True
    batch_size: int = 8
    max_new_tokens: int = 32
    max_prompt_tokens: int = 2048
    max_passage_words: int = 120
    prefix_k: int = 3

    retriever: str = "bm25"
    index_name: str = "wikipedia-dpr-100w"
    n_retrieve: int = 10
    top_k: int = 5
    search_threads: int = 8
    pyserini_cache_local: str = "/content/pyserini_cache"
    cached_retrieval_file: Optional[str] = None  # for retriever == "cached_jsonl"

    delta_thresholds: Tuple[float, ...] = (0.25, 0.5, 0.75)
    harm_thr: float = 0.5
    conf_bins: int = 5

    @property
    def model_tag(self) -> str:
        return safe(self.model_name)

    @property
    def retr_tag(self) -> str:
        return f"{safe(self.retriever)}_top{self.top_k}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["delta_thresholds"] = list(self.delta_thresholds)
        return d


def paths(cfg: Config) -> dict:
    w = cfg.workdir
    return {
        "data": os.path.join(w, "data"),
        "retrievals": os.path.join(w, "retrievals"),
        "logs": os.path.join(w, "logs", "gen"),
        "features": os.path.join(w, "features"),
        "results": os.path.join(w, "results", "phase1"),
        "figures": os.path.join(w, "figures", "phase1"),
        "pyserini_cache": os.path.join(w, "pyserini_cache"),
    }


def ensure_dirs(cfg: Config) -> None:
    for p in paths(cfg).values():
        os.makedirs(p, exist_ok=True)


def load_config(workdir: str, path: Optional[str] = None, **overrides) -> Config:
    """Load YAML (optional), apply keyword overrides, create the workdir layout."""
    d: dict = {}
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path) as f:
            d = yaml.safe_load(f) or {}
    d.update({k: v for k, v in overrides.items() if v is not None})
    d["workdir"] = workdir
    if "delta_thresholds" in d and d["delta_thresholds"] is not None:
        d["delta_thresholds"] = tuple(float(x) for x in d["delta_thresholds"])
    unknown = set(d) - set(Config.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = Config(**d)
    ensure_dirs(cfg)
    return cfg
