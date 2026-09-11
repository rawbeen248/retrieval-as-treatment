"""Join both arms' outcomes with pre-decision features into one table.

Feature tiers (see research plan §4.3):
  F0 — no retrieval side-effects: question stats, popularity, arm-0 probe stats.
  F1 — cheap index lookups allowed: retrieval score statistics (no passages read
       into the LLM). Named bm25_* here because Phase 1 uses BM25; Phase 2 adds
       dense_* columns.

Outcome columns: y{arm}_{f1,em,acc}; effect columns: delta_{f1,em,acc}.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .generate import PROBE_FIELDS

F0_COLS = ["q_len_words", "n_answers", "log_pop", "idk_flag", "ans_ntokens"] + PROBE_FIELDS
F1_COLS = ["retr_top1", "retr_mean_topk", "retr_std_topk", "retr_gap12"]


def _retrieval_feats(hits: List[dict], top_k: int) -> Dict[str, float]:
    s = [float(h.get("score", np.nan)) for h in (hits or [])[:top_k]]
    if not s:
        return {c: np.nan for c in F1_COLS}
    arr = np.asarray(s, dtype=float)
    return {
        "retr_top1": float(arr[0]),
        "retr_mean_topk": float(arr.mean()),
        "retr_std_topk": float(arr.std(ddof=0)),
        "retr_gap12": float(arr[0] - arr[1]) if len(arr) > 1 else np.nan,
    }


def build_feature_table(data_df: pd.DataFrame, gen0_df: pd.DataFrame, gen1_df: pd.DataFrame,
                        retr_by_qid: Dict[str, List[dict]], top_k: int) -> pd.DataFrame:
    keep = [c for c in ["qid", "dataset", "question", "answers", "s_pop", "o_pop", "log_pop", "pop_bin", "prop", "subj"]
            if c in data_df.columns]
    base = data_df[keep].copy()
    base["qid"] = base["qid"].astype(str)
    base["q_len_words"] = base["question"].astype(str).str.split().map(len)
    base["n_answers"] = base["answers"].map(lambda a: len(list(a)) if a is not None else 0)

    g0 = gen0_df.copy(); g0["qid"] = g0["qid"].astype(str); g0 = g0.set_index("qid")
    g1 = gen1_df.copy(); g1["qid"] = g1["qid"].astype(str); g1 = g1.set_index("qid")

    for arm, g in ((0, g0), (1, g1)):
        for m in ("f1", "em", "acc"):
            base[f"y{arm}_{m}"] = base["qid"].map(g[m]) if m in g.columns else np.nan
        base[f"answer{arm}"] = base["qid"].map(g["answer"]) if "answer" in g.columns else ""
    for c in PROBE_FIELDS + ["idk_flag", "ans_ntokens"]:
        base[c] = base["qid"].map(g0[c]) if c in g0.columns else np.nan
    if "n_passages_used" in g1.columns:
        base["n_passages_used"] = base["qid"].map(g1["n_passages_used"])

    rf = pd.DataFrame([_retrieval_feats(retr_by_qid.get(q, []), top_k) for q in base["qid"]], index=base.index)
    base = pd.concat([base, rf], axis=1)

    for m in ("f1", "em", "acc"):
        base[f"delta_{m}"] = base[f"y1_{m}"] - base[f"y0_{m}"]

    n_before = len(base)
    base = base.dropna(subset=["y0_f1", "y1_f1"]).reset_index(drop=True)
    if len(base) < n_before:
        print(f"[features] dropped {n_before - len(base)} queries missing one arm")
    return base
