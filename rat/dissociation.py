"""Which arm does each signal know about, and does that make it useful for the decision?

The decision quantity is tau(x) = mu1(x) - mu0(x), where
    mu0(x) = P(correct | no retrieval, x)     mu1(x) = P(correct | retrieval, x)

Two analyses, both non-circular because BOTH arms are observed for every query:

1. ARM DECOMPOSITION. For each pre-decision feature, AUC against Y0 and against
   Y1 separately, plus the signed separation D(f) = AUC(f -> Y1) - AUC(f -> Y0).
   Confidence features should own Y0 (D < 0); retrieval features should own Y1
   (D > 0). The dissociation statistic is D(retrieval) - D(confidence) via paired
   bootstrap.

2. TAU EVALUATION. Cross-fitted tau_hat = mu1_hat - mu0_hat from each feature set,
   scored by (a) correlation with the observed Delta and (b) realized policy value
   at matched retrieval rates. This is what a gate actually has to do.

METHODOLOGICAL NOTE -- two traps, both hit during development, both avoided here:
  * AUC(feature -> "helped") over ALL queries is contaminated: `helped` requires
    Y0 = 0, so any predictor of the level anti-predicts it for free.
  * AUC(feature -> "helped") on the DECISION-LIVE subset is worse: with binary
    outcomes, "the arms disagree" means exactly one is right, so `helped` is
    identically 1 - Y0 there. That target is the level in disguise.
  Neither is reported. Everything below is measured against Y0, Y1, or Delta.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CONF_FEATURES = ["probe_maxprob_mean", "probe_entropy_mean", "first_tok_maxprob", "ans_logprob_mean"]
POP_FEATURES = ["log_pop"]
AMBIG_FEATURES = ["subj_chars", "subj_ntokens", "subj_single_token", "q_len_words"]
RETR_FEATURES = ["retr_top1", "retr_top1_z_bylen", "retr_top1_resid", "retr_gap12", "retr_ratio_gap12",
                 "retr_ratio_mean", "retr_cv_topk", "retr_title_match_frac", "retr_subj_in_top1"]
DEFAULT_FEATURES = CONF_FEATURES + POP_FEATURES + AMBIG_FEATURES + RETR_FEATURES

TIER = {**{f: "F0-confidence" for f in CONF_FEATURES}, **{f: "F0-popularity" for f in POP_FEATURES},
        **{f: "F0-ambiguity" for f in AMBIG_FEATURES}, **{f: "F1-retrieval" for f in RETR_FEATURES}}

FEATURE_SETS = {
    "confidence only": CONF_FEATURES,
    "popularity only": POP_FEATURES,
    "ambiguity only": AMBIG_FEATURES,
    "retrieval only": RETR_FEATURES,
    "confidence + popularity": CONF_FEATURES + POP_FEATURES,
    "all F0 (no index lookup)": CONF_FEATURES + POP_FEATURES + AMBIG_FEATURES,
    "all F0 + F1": DEFAULT_FEATURES,
}


# --------------------------------------------------------------------- AUC
def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties handled. nan if degenerate."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(x).rank().to_numpy()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _boot_idx(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def boot_ci(values: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def boot_auc_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    point = auc(x, y)
    if not np.isfinite(point) or x.size < 10:
        return point, float("nan"), float("nan")
    idx = _boot_idx(x.size, n_boot, seed)
    lo, hi = boot_ci([auc(x[s], y[s]) for s in idx])
    return point, lo, hi


def _sep(a0: float, a1: float) -> float:
    """Orientation-invariant separation: how much more a feature knows about arm 1
    than arm 0. Uses |AUC - 0.5| so inverted features (entropy) score correctly."""
    if not (np.isfinite(a0) and np.isfinite(a1)):
        return float("nan")
    return abs(a1 - 0.5) - abs(a0 - 0.5)


# ------------------------------------------------- 1. arm decomposition
def arm_decomposition(feat: pd.DataFrame, features: Optional[Sequence[str]] = None,
                      n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per feature: AUC against each arm, and the signed separation D = AUC1 - AUC0."""
    features = [f for f in (features or DEFAULT_FEATURES) if f in feat.columns]
    y0 = feat["y0_acc"].to_numpy(dtype=float)
    y1 = feat["y1_acc"].to_numpy(dtype=float)
    rows = []
    for i, f in enumerate(features):
        x = feat[f].to_numpy(dtype=float)
        if np.isfinite(x).sum() < 10 or np.nanstd(x) == 0:
            continue
        a0, a0lo, a0hi = boot_auc_ci(x, y0, n_boot, seed + i)
        a1, a1lo, a1hi = boot_auc_ci(x, y1, n_boot, seed + 500 + i)
        m = np.isfinite(x) & np.isfinite(y0) & np.isfinite(y1)
        idx = _boot_idx(int(m.sum()), n_boot, seed + 1500 + i)
        xs, y0s, y1s = x[m], y0[m], y1[m]
        dlo, dhi = boot_ci([_sep(auc(xs[s], y0s[s]), auc(xs[s], y1s[s])) for s in idx])
        rows.append({
            "feature": f, "tier": TIER.get(f, "?"),
            "auc_Y0": a0, "Y0_lo": a0lo, "Y0_hi": a0hi,
            "auc_Y1": a1, "Y1_lo": a1lo, "Y1_hi": a1hi,
            "strength_Y0": abs(a0 - 0.5), "strength_Y1": abs(a1 - 0.5),
            "D": _sep(a0, a1), "D_lo": dlo, "D_hi": dhi,
            "knows_Y0": bool(np.isfinite(a0lo) and (a0lo > 0.5 or a0hi < 0.5)),
            "knows_Y1": bool(np.isfinite(a1lo) and (a1lo > 0.5 or a1hi < 0.5)),
            "separated": bool(np.isfinite(dlo) and (dlo > 0 or dhi < 0)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.reindex(out["D"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return out


def dissociation_statistic(feat: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap on D(best retrieval feature) - D(best confidence feature).

    Positive and CI excluding 0 means the two families genuinely attach to
    different arms -- the structural claim, with no circular target anywhere.
    """
    def strength(f, col):
        a = auc(feat[f].to_numpy(dtype=float), feat[col].to_numpy(dtype=float))
        return abs(a - 0.5) if np.isfinite(a) else -1.0

    conf = [f for f in CONF_FEATURES if f in feat.columns and feat[f].notna().sum() > 10]
    retr = [f for f in RETR_FEATURES if f in feat.columns and feat[f].notna().sum() > 10]
    if not conf or not retr:
        return {}
    c_best = max(conf, key=lambda f: strength(f, "y0_acc"))   # best at the parametric arm
    r_best = max(retr, key=lambda f: strength(f, "y1_acc"))   # best at the retrieval arm

    cols = [c_best, r_best, "y0_acc", "y1_acc"]
    d = feat[cols].to_numpy(dtype=float)
    d = d[np.isfinite(d).all(axis=1)]
    if d.shape[0] < 20:
        return {}
    xc, xr, y0, y1 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    D_conf = _sep(auc(xc, y0), auc(xc, y1))
    D_retr = _sep(auc(xr, y0), auc(xr, y1))
    idx = _boot_idx(d.shape[0], n_boot, seed)
    diffs = [_sep(auc(xr[s], y0[s]), auc(xr[s], y1[s])) - _sep(auc(xc[s], y0[s]), auc(xc[s], y1[s]))
             for s in idx]
    lo, hi = boot_ci(diffs)
    v = np.asarray([x for x in diffs if np.isfinite(x)])
    return {"conf_feature": c_best, "retr_feature": r_best, "n": int(d.shape[0]),
            "conf_auc_Y0": auc(xc, y0), "conf_auc_Y1": auc(xc, y1), "D_conf": D_conf,
            "retr_auc_Y0": auc(xr, y0), "retr_auc_Y1": auc(xr, y1), "D_retr": D_retr,
            "diff": float(D_retr - D_conf), "lo": lo, "hi": hi,
            "p_gt_0": float((v > 0).mean()) if v.size else float("nan")}


# ----------------------------------------------------- 2. tau evaluation
def crossfit_tau(feat: pd.DataFrame, features: Sequence[str], n_splits: int = 5, seed: int = 0,
                 C: float = 1.0) -> np.ndarray:
    """Cross-fitted tau_hat = mu1_hat - mu0_hat (logistic; the DR-learner nuisances
    in Phase 3 reuse this shape)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    cols = [c for c in features if c in feat.columns and feat[c].notna().sum() > 10 and feat[c].nunique() > 1]
    if not cols:
        return np.zeros(len(feat))
    X = feat[cols].astype(float)
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1.0)
    X = X.fillna(0.0).to_numpy()
    y0 = feat["y0_acc"].to_numpy(dtype=float)
    y1 = feat["y1_acc"].to_numpy(dtype=float)
    tau = np.zeros(len(feat))
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X):
        p = {}
        for name, y in (("0", y0), ("1", y1)):
            if len(np.unique(y[tr])) < 2:
                p[name] = np.full(len(te), float(y[tr].mean()))
            else:
                p[name] = LogisticRegression(max_iter=2000, C=C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        tau[te] = p["1"] - p["0"]
    return tau


def policy_value(feat: pd.DataFrame, score: np.ndarray, rate: float, cost: float = 0.0) -> float:
    """Value of retrieving for the top `rate` fraction by `score`."""
    n = len(feat)
    k = int(round(rate * n))
    R = np.zeros(n, dtype=bool)
    if k > 0:
        R[np.argsort(-np.asarray(score, dtype=float), kind="stable")[:k]] = True
    y0 = feat["y0_acc"].to_numpy(dtype=float)
    y1 = feat["y1_acc"].to_numpy(dtype=float)
    return float(np.where(R, y1 - cost, y0).mean())


def tau_evaluation(feat: pd.DataFrame, feature_sets: Optional[Dict[str, Sequence[str]]] = None,
                   rates: Sequence[float] = (0.25, 0.5, 0.75), n_boot: int = 1000,
                   seed: int = 0) -> Tuple[pd.DataFrame, dict]:
    """Score each feature set by how well its tau_hat drives the decision."""
    feature_sets = feature_sets or FEATURE_SETS
    delta = feat["delta_acc"].to_numpy(dtype=float)
    y0 = feat["y0_acc"].to_numpy(dtype=float)
    y1 = feat["y1_acc"].to_numpy(dtype=float)
    ref = {
        "never_retrieve": float(y0.mean()),
        "always_retrieve": float(y1.mean()),
        "oracle": float(np.where(delta > 0, y1, y0).mean()),
        "n": int(len(feat)),
    }
    ref["headroom_over_always"] = ref["oracle"] - ref["always_retrieve"]
    idx = _boot_idx(len(feat), n_boot, seed)

    rows = []
    for name, fs in feature_sets.items():
        cols = [c for c in fs if c in feat.columns and feat[c].notna().sum() > 10]
        if not cols:
            continue
        tau = crossfit_tau(feat, cols, seed=seed)
        if np.std(tau) == 0:
            continue
        r = {"feature_set": name, "n_features": len(cols),
             "corr_tau_delta": float(np.corrcoef(tau, delta)[0, 1])}
        cs = [float(np.corrcoef(tau[s], delta[s])[0, 1]) if np.std(tau[s]) > 0 and np.std(delta[s]) > 0 else np.nan
              for s in idx]
        r["corr_lo"], r["corr_hi"] = boot_ci(cs)
        for rate in rates:
            r[f"V@{int(100 * rate)}%"] = policy_value(feat, tau, rate)
        sub = feat.reset_index(drop=True)
        vs = [policy_value(sub.iloc[s], tau[s], 0.5) for s in idx[: min(n_boot, 400)]]
        r["V@50%_lo"], r["V@50%_hi"] = boot_ci(vs)
        rows.append(r)

    # single-feature proxy gates, for reference
    for label, col, sign in (("gate: 1 − confidence", "probe_maxprob_mean", -1.0),
                             ("gate: −popularity", "log_pop", -1.0),
                             ("gate: retrieval top-1", "retr_top1_z_bylen", 1.0)):
        if col in feat.columns and feat[col].notna().sum() > 10:
            s = sign * feat[col].fillna(feat[col].median()).to_numpy(dtype=float)
            r = {"feature_set": label, "n_features": 1,
                 "corr_tau_delta": float(np.corrcoef(s, delta)[0, 1]) if np.std(s) > 0 else np.nan}
            r["corr_lo"] = r["corr_hi"] = np.nan
            for rate in rates:
                r[f"V@{int(100 * rate)}%"] = policy_value(feat, s, rate)
            r["V@50%_lo"] = r["V@50%_hi"] = np.nan
            rows.append(r)

    out = pd.DataFrame(rows).sort_values("V@50%", ascending=False).reset_index(drop=True)
    return out, ref


def regime_diagnostic(feat: pd.DataFrame, arms: pd.DataFrame, tau_tbl: pd.DataFrame) -> dict:
    """Which arm carries the variance, and therefore which family of signals matters.

    tau = mu1 - mu0. When the model is at a floor (mu0 near 0 and flat), tau is
    driven by mu1 and retrieval-side signals win. When the model is competent,
    mu0 varies and confidence wins. This is the regime claim.
    """
    y0m, y1m = float(feat["y0_acc"].mean()), float(feat["y1_acc"].mean())
    def best(tier, col):
        s = arms[arms["tier"] == tier]
        if s.empty:
            return float("nan")
        return float((s[col] - 0.5).abs().max() + 0.5)
    conf_pred = best("F0-confidence", "auc_Y0")
    retr_pred = best("F1-retrieval", "auc_Y1")
    def v(name):
        r = tau_tbl[tau_tbl["feature_set"] == name]
        return float(r["V@50%"].iloc[0]) if len(r) else float("nan")
    v_conf, v_retr = v("confidence only"), v("retrieval only")
    winner = ("confidence" if v_conf > v_retr else "retrieval") if np.isfinite(v_conf) and np.isfinite(v_retr) else "?"
    return {"Y0_mean": y0m, "Y1_mean": y1m,
            "arm0_variance_room": float(y0m * (1 - y0m)), "arm1_variance_room": float(y1m * (1 - y1m)),
            "best_conf_auc_Y0": conf_pred, "best_retr_auc_Y1": retr_pred,
            "V50_confidence_only": v_conf, "V50_retrieval_only": v_retr,
            "winner_at_50pct": winner,
            "regime": "model-limited (arm 0 near floor)" if y0m < 0.30 else
                      ("corpus-limited (arm 0 competent)" if y0m > 0.45 else "mixed")}
