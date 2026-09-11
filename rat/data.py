"""Dataset loading into one unified schema.

Columns: qid, dataset, question, answers (list[str]), s_pop, o_pop, log_pop,
pop_bin, prop, subj.  Non-PopQA datasets carry NaN popularity and pop_bin = -1.

PopQA is sampled *stratified by popularity* on purpose: a uniform sample is
dominated by long-tail entities, and the harm map needs the popular end
(where retrieval is most likely to hurt) to be well populated.
"""
from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np
import pandas as pd


def _parse_answers(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(a) for a in x if str(a).strip()]
    if isinstance(x, str):
        s = x.strip()
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(a) for a in v if str(a).strip()]
        except json.JSONDecodeError:
            pass
        return [s] if s else []
    return [str(x)]


def _first_split(d):
    """datasets.load_dataset without split= returns a DatasetDict."""
    try:
        keys = list(d.keys())
        return d[keys[0]]
    except AttributeError:
        return d


# -------------------------------------------------------------------- PopQA
def load_popqa() -> pd.DataFrame:
    from datasets import load_dataset

    last = None
    ds = None
    for name, kw in (("akariasai/PopQA", {"split": "test"}), ("akariasai/PopQA", {})):
        try:
            d = load_dataset(name, **kw)
            ds = d if kw else _first_split(d)
            break
        except Exception as e:  # noqa: BLE001
            last = e
    if ds is None:
        raise RuntimeError(f"Could not load PopQA from the Hub: {last}")
    df = ds.to_pandas()

    def col(name, default=np.nan):
        return df[name] if name in df.columns else pd.Series([default] * len(df))

    ids = df["id"].astype(str) if "id" in df.columns else pd.Series(range(len(df))).astype(str)
    out = pd.DataFrame(
        {
            "qid": "popqa_" + ids,
            "dataset": "popqa",
            "question": df["question"].astype(str),
            "answers": [_parse_answers(a) for a in col("possible_answers", None)],
            "s_pop": pd.to_numeric(col("s_pop"), errors="coerce"),
            "o_pop": pd.to_numeric(col("o_pop"), errors="coerce"),
            "prop": col("prop", "").astype(str),
            "subj": col("subj", "").astype(str),
        }
    )
    out = out[out["answers"].map(len) > 0].reset_index(drop=True)
    return out


# ------------------------------------------------------------------ NQ-open
def load_nq_open() -> pd.DataFrame:
    from datasets import load_dataset

    last = None
    ds = None
    for name in ("google-research-datasets/nq_open", "nq_open"):
        try:
            ds = load_dataset(name, split="validation")
            break
        except Exception as e:  # noqa: BLE001
            last = e
    if ds is None:
        raise RuntimeError(f"Could not load NQ-open: {last}")
    df = ds.to_pandas()
    out = pd.DataFrame(
        {
            "qid": [f"nq_{i}" for i in range(len(df))],
            "dataset": "nq_open",
            "question": df["question"].astype(str),
            "answers": [_parse_answers(a) for a in df["answer"]],
            "s_pop": np.nan,
            "o_pop": np.nan,
            "prop": "",
            "subj": "",
        }
    )
    return out[out["answers"].map(len) > 0].reset_index(drop=True)


# ----------------------------------------------------------------- TriviaQA
def load_triviaqa() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    rows = []
    for i, ex in enumerate(ds):
        ans = ex.get("answer", {}) or {}
        aliases = list(ans.get("aliases", []) or []) + [ans.get("value", "")]
        rows.append(
            {
                "qid": f"tqa_{ex.get('question_id', i)}",
                "dataset": "triviaqa",
                "question": str(ex["question"]),
                "answers": [a for a in dict.fromkeys(aliases) if a],
                "s_pop": np.nan,
                "o_pop": np.nan,
                "prop": "",
                "subj": "",
            }
        )
    out = pd.DataFrame(rows)
    return out[out["answers"].map(len) > 0].reset_index(drop=True)


LOADERS = {
    "popqa": load_popqa,
    "nq_open": load_nq_open,
    "nq": load_nq_open,
    "triviaqa": load_triviaqa,
    "trivia_qa": load_triviaqa,
}


# ----------------------------------------------------------------- Sampling
def sample_stratified(df: pd.DataFrame, n: int, seed: int, pop_bins: int) -> Tuple[pd.DataFrame, dict]:
    """Equal-size sample from popularity quantile bins (if popularity exists).

    Returns (sample, meta) where meta records the bin edges used so the harm
    map rows can be labelled with actual popularity ranges.
    """
    df = df.copy()
    meta: dict = {"n_requested": int(n), "seed": int(seed)}
    has_pop = df["s_pop"].notna().mean() > 0.5

    if has_pop:
        df["log_pop"] = np.log10(df["s_pop"].fillna(0).clip(lower=0) + 1.0)
        codes, edges = pd.qcut(df["log_pop"], q=pop_bins, labels=False, retbins=True, duplicates="drop")
        df["pop_bin"] = codes.astype(int)
        nb = int(df["pop_bin"].nunique())
        per = max(1, n // nb)
        parts = [g.sample(n=min(per, len(g)), random_state=seed) for _, g in df.groupby("pop_bin")]
        out = pd.concat(parts)
        if len(out) < n:  # top up from the remainder, if any strata were short
            rest = df.drop(out.index)
            out = pd.concat([out, rest.sample(n=min(n - len(out), len(rest)), random_state=seed)])
        meta.update(
            {
                "stratified": True,
                "pop_bins": nb,
                "pop_bin_edges_log10": [float(e) for e in edges],
                "per_bin_counts": out["pop_bin"].value_counts().sort_index().to_dict(),
            }
        )
    else:
        df["log_pop"] = np.nan
        df["pop_bin"] = -1
        out = df.sample(n=min(n, len(df)), random_state=seed)
        meta.update({"stratified": False})

    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    meta["n_actual"] = int(len(out))
    return out, meta


def load_unified(name: str, n: int, seed: int, pop_bins: int = 5) -> Tuple[pd.DataFrame, dict]:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset {name!r}; known: {sorted(LOADERS)}")
    full = LOADERS[name]()
    sample, meta = sample_stratified(full, n, seed, pop_bins)
    meta["dataset"] = name
    meta["n_full"] = int(len(full))
    return sample, meta
