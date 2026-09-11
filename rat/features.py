"""Feature table: both arms' outcomes joined with pre-decision features.

Feature tiers (research plan §4.3). Revised after the Phase-1 pilot, which found
that LLM confidence predicts the *level* but not the *effect*, while entity
surface form and retrieval scores predict the *effect*.

  F0  no retrieval side-effects at all
      question stats, popularity, arm-0 probe stats,
      AMBIGUITY: subject length / token count / single-token flag
  F1  cheap index lookups allowed (search, but no passages read into context)
      raw BM25 scores (NOT comparable across queries -- kept for reference only),
      NORMALIZED retrieval features (rank ratios, dispersion ratios, z within
      query-length stratum), and corpus-side ambiguity (how many distinct titles
      in the top-k actually match the subject string).

The pilot showed raw retr_top1 correlates r ~ 0.70 with subject-name length,
because longer names put more IDF mass in the query. `retr_top1_resid` and the
ratio features exist so the paper can separate "the retriever found something"
from "the query happened to be long".
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .generate import PROBE_FIELDS

# tier F0
F0_CONF = list(PROBE_FIELDS) + ["idk_flag", "ans_ntokens"]
F0_AMBIG = ["subj_chars", "subj_ntokens", "subj_single_token", "subj_is_short"]
F0_QUERY = ["q_len_words", "n_answers", "log_pop"]
F0_COLS = F0_QUERY + F0_AMBIG + F0_CONF

# tier F1
F1_RAW = ["retr_top1", "retr_mean_topk", "retr_std_topk", "retr_gap12"]
F1_NORM = [
    "retr_ratio_gap12",       # (s1 - s2) / s1        : is the top hit distinctive?
    "retr_ratio_mean",        # mean(s_1..k) / s1     : does the tail keep up?
    "retr_cv_topk",           # std / mean            : scale-free dispersion
    "retr_top1_z_bylen",      # s1 z-scored within query-length stratum
    "retr_top1_resid",        # s1 residualized on subj_chars + q_len_words
    "retr_title_match_frac",  # frac of top-k titles containing the subject
    "retr_title_nunique",     # distinct titles in top-k
    "retr_subj_in_top1",      # subject string appears in top-1 title
]
F1_COLS = F1_RAW + F1_NORM

_CAP_SPAN = re.compile(r"\b([A-Z][\w'’\-]*(?:\s+(?:of|the|de|van|von|da|di)\s+[A-Z\w'’\-]+|\s+[A-Z][\w'’\-]*)*)")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower()).strip()


def subject_of(row) -> str:
    """PopQA ships `subj`. Other datasets don't, so fall back to the longest
    capitalized span in the question (a rough entity proxy -- flagged as such
    in the paper, and only used for ambiguity features)."""
    s = str(row.get("subj", "") or "").strip()
    if s:
        return s
    q = str(row.get("question", ""))
    q = re.sub(r"^(who|what|where|when|which|in what|how)\b", "", q, flags=re.I)
    spans = [m.group(1).strip() for m in _CAP_SPAN.finditer(q)]
    return max(spans, key=len) if spans else ""


def ambiguity_feats(subject: str) -> Dict[str, float]:
    s = str(subject or "").strip()
    ntok = len(s.split())
    return {
        "subj_chars": float(len(s)),
        "subj_ntokens": float(ntok),
        "subj_single_token": float(ntok <= 1),
        "subj_is_short": float(len(s) <= 8),   # pilot: every genuine harm case had <= 6 chars
    }


def _retrieval_feats(hits: List[dict], top_k: int, subject: str) -> Dict[str, float]:
    out: Dict[str, float] = {c: np.nan for c in F1_RAW + F1_NORM}
    hits = (hits or [])[:top_k]
    if not hits:
        return out
    s = np.asarray([float(h.get("score", np.nan)) for h in hits], dtype=float)
    s = s[np.isfinite(s)]
    if s.size:
        out["retr_top1"] = float(s[0])
        out["retr_mean_topk"] = float(s.mean())
        out["retr_std_topk"] = float(s.std(ddof=0))
        out["retr_gap12"] = float(s[0] - s[1]) if s.size > 1 else np.nan
        denom = abs(s[0]) if abs(s[0]) > 1e-9 else np.nan
        out["retr_ratio_gap12"] = float((s[0] - s[1]) / denom) if s.size > 1 else np.nan
        out["retr_ratio_mean"] = float(s.mean() / denom)
        out["retr_cv_topk"] = float(s.std(ddof=0) / abs(s.mean())) if abs(s.mean()) > 1e-9 else np.nan

    nsubj = _norm(subject)
    titles = [str(h.get("title", "")) for h in hits]
    ntitles = [_norm(t) for t in titles]
    if nsubj:
        match = [1.0 for t in ntitles if nsubj and nsubj in t]
        out["retr_title_match_frac"] = float(len(match) / max(len(ntitles), 1))
        out["retr_subj_in_top1"] = float(bool(ntitles and nsubj in ntitles[0]))
    out["retr_title_nunique"] = float(len(set(t for t in ntitles if t)))
    return out


def _add_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-query normalizations (need the whole table, so done after the join)."""
    if "retr_top1" not in df.columns or df["retr_top1"].notna().sum() < 10:
        df["retr_top1_z_bylen"] = np.nan
        df["retr_top1_resid"] = np.nan
        return df

    # z-score within query-length stratum
    try:
        strat = pd.qcut(df["q_len_words"], q=min(4, df["q_len_words"].nunique()), labels=False, duplicates="drop")
    except ValueError:
        strat = pd.Series(0, index=df.index)
    g = df.groupby(strat)["retr_top1"]
    mu, sd = g.transform("mean"), g.transform("std").replace(0, np.nan)
    df["retr_top1_z_bylen"] = (df["retr_top1"] - mu) / sd

    # residualize on subject length + question length (the pilot's confound)
    cols = ["subj_chars", "q_len_words"]
    ok = df[cols + ["retr_top1"]].notna().all(axis=1)
    df["retr_top1_resid"] = np.nan
    if ok.sum() >= 10:
        X = np.c_[np.ones(ok.sum()), df.loc[ok, cols].to_numpy(dtype=float)]
        y = df.loc[ok, "retr_top1"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        df.loc[ok, "retr_top1_resid"] = y - X @ beta
    return df


def build_feature_table(data_df: pd.DataFrame, gen0_df: pd.DataFrame, gen1_df: pd.DataFrame,
                        retr_by_qid: Dict[str, List[dict]], top_k: int) -> pd.DataFrame:
    keep = [c for c in ["qid", "dataset", "question", "answers", "s_pop", "o_pop", "log_pop", "pop_bin", "prop", "subj"]
            if c in data_df.columns]
    base = data_df[keep].copy()
    base["qid"] = base["qid"].astype(str)
    base["q_len_words"] = base["question"].astype(str).str.split().map(len)
    base["n_answers"] = base["answers"].map(lambda a: len(list(a)) if a is not None else 0)
    if "log_pop" not in base.columns:
        base["log_pop"] = np.nan

    # ambiguity (F0)
    base["subject"] = [subject_of(r) for r in base.to_dict("records")]
    amb = pd.DataFrame([ambiguity_feats(s) for s in base["subject"]], index=base.index)
    base = pd.concat([base, amb], axis=1)

    g0 = gen0_df.copy(); g0["qid"] = g0["qid"].astype(str); g0 = g0.set_index("qid")
    g1 = gen1_df.copy(); g1["qid"] = g1["qid"].astype(str); g1 = g1.set_index("qid")
    for arm, g in ((0, g0), (1, g1)):
        for m in ("f1", "em", "acc"):
            base[f"y{arm}_{m}"] = base["qid"].map(g[m]) if m in g.columns else np.nan
        base[f"answer{arm}"] = base["qid"].map(g["answer"]) if "answer" in g.columns else ""
    for c in F0_CONF:
        base[c] = base["qid"].map(g0[c]) if c in g0.columns else np.nan
    if "n_passages_used" in g1.columns:
        base["n_passages_used"] = base["qid"].map(g1["n_passages_used"])

    rf = pd.DataFrame(
        [_retrieval_feats(retr_by_qid.get(q, []), top_k, s) for q, s in zip(base["qid"], base["subject"])],
        index=base.index,
    )
    base = pd.concat([base, rf], axis=1)
    base = _add_normalized(base)

    for m in ("f1", "em", "acc"):
        base[f"delta_{m}"] = base[f"y1_{m}"] - base[f"y0_{m}"]
    # primary effect labels are containment-based (pilot: F1 over-counts harm ~2x
    # via partial credit and extra-token penalties)
    base["helped"] = (base["delta_acc"] > 0).astype(float)
    base["harmed"] = (base["delta_acc"] < 0).astype(float)
    base["both_wrong"] = ((base["y0_acc"] == 0) & (base["y1_acc"] == 0)).astype(float)
    base["both_right"] = ((base["y0_acc"] == 1) & (base["y1_acc"] == 1)).astype(float)

    n_before = len(base)
    base = base.dropna(subset=["y0_f1", "y1_f1"]).reset_index(drop=True)
    if len(base) < n_before:
        print(f"[features] dropped {n_before - len(base)} queries missing one arm")
    return base
