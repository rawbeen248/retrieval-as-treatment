"""The level-vs-effect dissociation -- the paper's headline analysis.

Every published adaptive-retrieval gate thresholds a proxy for the LEVEL
("will the parametric answer be wrong?"). The decision-relevant quantity is the
EFFECT ("will retrieving change the answer for the better?"). If a signal
predicts one and not the other, gating on it is provably not the optimal rule.

For each pre-decision feature we report, with bootstrap CIs:
    AUC_level      = AUC( feature -> Y0 correct )
    AUC_effect     = AUC( feature -> retrieval helped ), over all queries
    AUC_effect_live= AUC( feature -> helped rather than harmed ), over the queries
                     where the arms actually DISAGREE (delta != 0)
    gap            = AUC_effect_live - AUC_level

Why the third column exists (and is the honest one): `helped` requires Y0 to be
wrong, so ANY predictor of the level mechanically anti-predicts `helped` whether
or not it knows anything about retrieval. Reading AUC_effect alone would credit
confidence with "effect information" that is pure arithmetic. Restricting to the
decision-live subset -- arms disagree, so exactly one of helped/harmed is true --
removes that coupling and asks the question a gate actually faces: given that
retrieval will change the answer, will it change it for the better?

AUCs are made direction-free by taking max(a, 1-a) is NOT used -- direction
matters for a gate, so we report the raw AUC and a `flip` column telling you
which way the feature points.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_FEATURES = [
    # F0 confidence (what the literature gates on)
    "probe_maxprob_mean", "probe_entropy_mean", "first_tok_maxprob", "ans_logprob_mean",
    # F0 popularity (Mallen et al.)
    "log_pop",
    # F0 ambiguity (pilot finding)
    "subj_chars", "subj_ntokens", "subj_single_token", "q_len_words",
    # F1 retrieval, raw then normalized
    "retr_top1", "retr_gap12", "retr_ratio_gap12", "retr_ratio_mean", "retr_cv_topk",
    "retr_top1_z_bylen", "retr_top1_resid", "retr_title_match_frac", "retr_subj_in_top1",
]

TIER = {
    **{f: "F0-confidence" for f in ["probe_maxprob_mean", "probe_entropy_mean", "first_tok_maxprob", "ans_logprob_mean"]},
    **{f: "F0-popularity" for f in ["log_pop"]},
    **{f: "F0-ambiguity" for f in ["subj_chars", "subj_ntokens", "subj_single_token", "q_len_words"]},
    **{f: "F1-retrieval" for f in ["retr_top1", "retr_gap12", "retr_ratio_gap12", "retr_ratio_mean", "retr_cv_topk",
                                   "retr_top1_z_bylen", "retr_top1_resid", "retr_title_match_frac", "retr_subj_in_top1"]},
}


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties handled. Returns nan if degenerate."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    pos, neg = y > 0.5, y <= 0.5
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(x).rank().to_numpy()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0,
                alpha: float = 0.05) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    point = auc(x, y)
    if not np.isfinite(point):
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = np.arange(x.size)
    vals: List[float] = []
    for _ in range(n_boot):
        s = rng.choice(idx, idx.size, replace=True)
        a = auc(x[s], y[s])
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def live_mask(feat: pd.DataFrame, delta_col: str = "delta_acc") -> np.ndarray:
    """Queries where the two arms disagree -- the only ones a gate can affect."""
    d = feat[delta_col].to_numpy(dtype=float)
    return np.isfinite(d) & (d != 0)


def dissociation_table(feat: pd.DataFrame, features: Optional[Sequence[str]] = None,
                       level_col: str = "y0_acc", effect_col: str = "helped",
                       n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per feature: AUC for the level, for the effect, and for the effect
    restricted to the decision-live subset (see module docstring)."""
    features = [f for f in (features or DEFAULT_FEATURES) if f in feat.columns]
    y_level = feat[level_col].to_numpy(dtype=float)
    y_eff = feat[effect_col].to_numpy(dtype=float)
    live = live_mask(feat)
    n_live = int(live.sum())
    rows = []
    for i, f in enumerate(features):
        x = feat[f].to_numpy(dtype=float)
        if np.isfinite(x).sum() < 10 or np.nanstd(x) == 0:
            continue
        al, al_lo, al_hi = boot_auc_ci(x, y_level, n_boot, seed + i)
        ae, ae_lo, ae_hi = boot_auc_ci(x, y_eff, n_boot, seed + 1000 + i)
        if n_live >= 10:
            av, av_lo, av_hi = boot_auc_ci(x[live], y_eff[live], n_boot, seed + 2000 + i)
        else:
            av = av_lo = av_hi = float("nan")
        rows.append({
            "feature": f, "tier": TIER.get(f, "?"),
            "auc_level": al, "level_lo": al_lo, "level_hi": al_hi,
            "auc_effect": ae, "effect_lo": ae_lo, "effect_hi": ae_hi,
            "auc_effect_live": av, "live_lo": av_lo, "live_hi": av_hi, "n_live": n_live,
            "gap": (av - al) if np.isfinite(av) else (ae - al),
            "effect_informative": bool(np.isfinite(av_lo) and (av_lo > 0.5 or av_hi < 0.5)),
            "level_informative": bool(np.isfinite(al_lo) and (al_lo > 0.5 or al_hi < 0.5)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        key = "auc_effect_live" if out["auc_effect_live"].notna().any() else "auc_effect"
        out["_s"] = (out[key] - 0.5).abs()
        out = out.sort_values("_s", ascending=False).drop(columns="_s").reset_index(drop=True)
    return out


def boot_auc_diff(feat: pd.DataFrame, f_a: str, f_b: str, target: str = "helped",
                  n_boot: int = 2000, seed: int = 0, live_only: bool = False) -> dict:
    """Paired bootstrap on AUC(f_a) - AUC(f_b) for the same target.

    Paired (same resampled rows for both features), which is the right test for
    'does retrieval score beat confidence at predicting the effect?'.
    """
    if live_only:
        feat = feat[live_mask(feat)]
    d = feat[[f_a, f_b, target]].to_numpy(dtype=float)
    d = d[np.isfinite(d).all(axis=1)]
    if d.shape[0] < 10:
        return {"n": int(d.shape[0]), "diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_gt_0": float("nan")}
    xa, xb, y = d[:, 0], d[:, 1], d[:, 2]
    point = auc(xa, y) - auc(xb, y)
    rng = np.random.default_rng(seed)
    idx = np.arange(d.shape[0])
    vals = []
    for _ in range(n_boot):
        s = rng.choice(idx, idx.size, replace=True)
        a, b = auc(xa[s], y[s]), auc(xb[s], y[s])
        if np.isfinite(a) and np.isfinite(b):
            vals.append(a - b)
    if not vals:
        return {"n": int(d.shape[0]), "diff": point, "lo": float("nan"), "hi": float("nan"), "p_gt_0": float("nan")}
    v = np.asarray(vals)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"n": int(d.shape[0]), "feature_a": f_a, "feature_b": f_b,
            "target": target + (" (live only)" if live_only else ""),
            "diff": float(point), "lo": float(lo), "hi": float(hi), "p_gt_0": float((v > 0).mean())}


def headline_contrast(feat: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """The paper's one-sentence claim, computed.

    Best confidence feature vs best retrieval feature, on the effect and on the level.
    """
    conf = [f for f in ["probe_maxprob_mean", "probe_entropy_mean", "first_tok_maxprob"] if f in feat.columns]
    retr = [f for f in ["retr_top1_z_bylen", "retr_top1", "retr_ratio_gap12", "retr_title_match_frac"] if f in feat.columns]
    if not conf or not retr:
        return {}

    live = live_mask(feat)
    sub = feat[live] if live.sum() >= 10 else feat

    def best(cands):
        scored = [(abs(auc(sub[c].to_numpy(dtype=float), sub["helped"].to_numpy(dtype=float)) - 0.5), c)
                  for c in cands]
        scored = [(s, c) for s, c in scored if np.isfinite(s)]
        return max(scored)[1] if scored else None

    c_best, r_best = best(conf), best(retr)
    if c_best is None or r_best is None:
        return {}
    out = {"conf_feature": c_best, "retr_feature": r_best, "n_live": int(live.sum())}
    for name, f in (("conf", c_best), ("retr", r_best)):
        p, lo, hi = boot_auc_ci(feat[f].to_numpy(dtype=float), feat["y0_acc"].to_numpy(dtype=float), n_boot, seed)
        out[f"{name}_level"] = {"auc": p, "lo": lo, "hi": hi}
        p, lo, hi = boot_auc_ci(sub[f].to_numpy(dtype=float), sub["helped"].to_numpy(dtype=float), n_boot, seed)
        out[f"{name}_effect"] = {"auc": p, "lo": lo, "hi": hi}
    out["diff_on_effect"] = boot_auc_diff(feat, r_best, c_best, "helped", n_boot, seed,
                                          live_only=bool(live.sum() >= 10))
    out["diff_on_level"] = boot_auc_diff(feat, c_best, r_best, "y0_acc", n_boot, seed)
    return out
