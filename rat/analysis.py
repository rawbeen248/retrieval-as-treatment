"""Phase 1 analysis.

Restructured after the n=300 x 2 run:
  1. HEADLINE  arm decomposition + tau evaluation (rat/dissociation.py)
  2. SUPPORT   effect maps -- the informative axes (retrieval score x ambiguity)
               AND the literature's axes (confidence x popularity) as a contrast
  3. SUPPORT   flip table, harm audit, proxy-gate regret decomposition

Definitions:
  delta_m     = Y1_m - Y0_m for m in {f1, em, acc}
  helped      = delta_acc > 0        harmed = delta_acc < 0   (containment-based;
                the pilot showed token-F1 over-counts harm ~2x through partial
                credit and extra-token penalties)
  proxy gate  : retrieve iff score(q) >= t
  oracle(c)   : retrieve iff delta > c
  regret      = V(oracle) - V(gate) = harm_incurred + benefit_forgone   (exact)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .dissociation import (  # noqa: E402
    arm_decomposition, dissociation_statistic, regime_diagnostic, tau_evaluation,
)

OUTCOMES = ("f1", "em", "acc")
MIN_CELL_N = 20  # pilot lesson: a "0.33 harm rate" from n=3 is noise, not a finding


# ----------------------------------------------------------------- binning
def quantile_bins(x: pd.Series, n: int) -> Tuple[pd.Series, np.ndarray]:
    codes, edges = pd.qcut(x, q=n, labels=False, retbins=True, duplicates="drop")
    return codes, edges


def _bin_col(f: pd.DataFrame, src: str, dst: str, n: int, meta: dict) -> None:
    f[dst] = -1
    if src not in f.columns:
        return
    ok = f[src].notna()
    if ok.sum() >= n and f.loc[ok, src].nunique() >= 2:
        try:
            codes, edges = quantile_bins(f.loc[ok, src], n)
            f.loc[ok, dst] = codes.astype(int).values
            meta[f"{dst}_edges"] = [float(e) for e in edges]
            meta[f"{dst}_source"] = src
        except ValueError:
            pass


def add_bins(feat: pd.DataFrame, conf_col: str, n_conf: int, n_pop: int) -> Tuple[pd.DataFrame, dict]:
    f = feat.copy()
    meta: dict = {"conf_col": conf_col}
    _bin_col(f, conf_col, "conf_bin", n_conf, meta)
    if "pop_bin" not in f.columns or not (f["pop_bin"] >= 0).any():
        _bin_col(f, "log_pop", "pop_bin", n_pop, meta)
    elif "log_pop" in f.columns:
        meta["pop_bin_ranges_log10"] = f.groupby("pop_bin")["log_pop"].agg(["min", "max"]).round(2).to_dict("index")
    if "pop_bin" not in f.columns:
        f["pop_bin"] = -1
    # the axes the pilot says actually carry signal
    _bin_col(f, "retr_top1_z_bylen" if "retr_top1_z_bylen" in f.columns else "retr_top1", "retr_bin", n_conf, meta)
    _bin_col(f, "subj_chars", "ambig_bin", n_conf, meta)
    return f, meta


# ------------------------------------------------------------- effect maps
def cell_tables(feat: pd.DataFrame, row: str, col: str, delta_col: str = "delta_acc") -> Dict[str, pd.DataFrame]:
    d = feat.dropna(subset=[delta_col])
    d = d[(d[row] >= 0) & (d[col] >= 0)]
    if d.empty:
        empty = pd.DataFrame()
        return {"harmful": empty, "helpful": empty, "mean": empty, "n": empty}
    g = d.groupby([row, col])[delta_col]
    return {
        "harmful": g.apply(lambda s: float((s < 0).mean())).unstack(col),
        "helpful": g.apply(lambda s: float((s > 0).mean())).unstack(col),
        "mean": g.mean().unstack(col),
        "n": g.size().unstack(col),
    }


def marginal(feat: pd.DataFrame, by: str, delta_col: str = "delta_acc") -> pd.DataFrame:
    d = feat.dropna(subset=[delta_col])
    d = d[d[by] >= 0]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(by)[delta_col]
    size = g.size()
    out = pd.DataFrame({
        "n": size,
        "mean_delta": g.mean(),
        "se": g.std(ddof=1) / np.sqrt(size.clip(lower=1)),
        "help_rate": g.apply(lambda s: float((s > 0).mean())),
        "harm_rate": g.apply(lambda s: float((s < 0).mean())),
    })
    return out.reset_index()


def flip_table(feat: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(feat["y0_acc"].astype(int), feat["y1_acc"].astype(int))
    t.index = [f"arm0 wrong", f"arm0 right"][: len(t)]
    t.columns = [f"arm1 wrong", f"arm1 right"][: t.shape[1]]
    return t


def harm_audit(feat: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    """Every query with negative delta_f1, labelled by whether it is a genuine
    right->wrong flip or a scoring artifact. The pilot found >half were artifacts."""
    d = feat[feat["delta_f1"] < 0].sort_values("delta_f1")
    rows = []
    for _, r in d.iterrows():
        genuine = (r["y0_acc"] == 1) and (r["y1_acc"] == 0)
        if genuine:
            kind = "genuine flip"
        elif r["y1_acc"] == 1:
            kind = "artifact: arm1 still contains gold (extra-token penalty)"
        elif r["y0_acc"] == 0:
            kind = "artifact: both arms wrong (partial credit)"
        else:
            kind = "other"
        rows.append({
            "delta_f1": round(float(r["delta_f1"]), 3), "kind": kind,
            "question": str(r["question"])[:70],
            "gold": "; ".join(map(str, list(r["answers"])[:2]))[:45],
            "arm0": str(r.get("answer0", ""))[:38], "arm1": str(r.get("answer1", ""))[:38],
            "subj_chars": int(r.get("subj_chars", 0)),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- gate curves
def gate_curve(feat: pd.DataFrame, score_col: str, y0_col: str, y1_col: str,
               higher_means_retrieve: bool = True, cost: float = 0.0, n_points: int = 41) -> pd.DataFrame:
    d = feat.dropna(subset=[score_col, y0_col, y1_col])
    s = d[score_col].to_numpy(dtype=float)
    if not higher_means_retrieve:
        s = -s
    y0 = d[y0_col].to_numpy(dtype=float)
    y1 = d[y1_col].to_numpy(dtype=float)
    delta = y1 - y0
    v_oracle = float(np.where(delta > cost, y1 - cost, y0).mean())
    v_always = float((y1 - cost).mean())
    v_never = float(y0.mean())
    thresholds = np.unique(np.concatenate([[-np.inf], np.quantile(s, np.linspace(0, 1, n_points)), [np.inf]]))
    rows = []
    for t in thresholds:
        R = s >= t
        rate = float(R.mean())
        rows.append({
            "threshold": float(t), "retrieval_rate": rate,
            "value": float(np.where(R, y1 - cost, y0).mean()),
            "random_value": rate * v_always + (1 - rate) * v_never,
            "oracle_value": v_oracle,
            "regret": v_oracle - float(np.where(R, y1 - cost, y0).mean()),
            "harm_incurred": float((R * np.clip(cost - delta, 0, None)).mean()),
            "benefit_forgone": float(((~R) * np.clip(delta - cost, 0, None)).mean()),
        })
    out = pd.DataFrame(rows).sort_values("retrieval_rate").reset_index(drop=True)
    out.attrs.update({"v_oracle": v_oracle, "v_always": v_always, "v_never": v_never,
                      "n": int(len(d)), "cost": cost, "score_col": score_col})
    return out


def curve_at_rates(curve: pd.DataFrame, rates=(0.1, 0.25, 0.5, 0.75, 0.9)) -> pd.DataFrame:
    idx = [int((curve["retrieval_rate"] - r).abs().idxmin()) for r in rates]
    return curve.loc[idx].drop_duplicates().reset_index(drop=True)


def auc_of_curve(curve: pd.DataFrame) -> float:
    c = curve.sort_values("retrieval_rate")
    return float(np.trapezoid(c["value"].to_numpy(), c["retrieval_rate"].to_numpy()))


# ---------------------------------------------------------------- summary
def summary_stats(feat: pd.DataFrame, thresholds) -> dict:
    s: dict = {}
    for m in OUTCOMES:
        dcol = f"delta_{m}"
        if dcol not in feat.columns:
            continue
        d = feat[dcol].dropna()
        s[m] = {
            "n": int(len(d)),
            "y0_mean": float(feat[f"y0_{m}"].mean()),
            "y1_mean": float(feat[f"y1_{m}"].mean()),
            "delta_mean": float(d.mean()),
            "delta_se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
            "harm_rate": {str(t): float((d < -t).mean()) for t in thresholds},
            "help_rate": {str(t): float((d > t).mean()) for t in thresholds},
        }
    s["flips"] = {
        "helped": int(feat["helped"].sum()),
        "harmed": int(feat["harmed"].sum()),
        "both_right": int(feat["both_right"].sum()),
        "both_wrong": int(feat["both_wrong"].sum()),
        "decision_live_frac": float(1 - feat["both_wrong"].mean() - feat["both_right"].mean()),
    }
    return s


def decision_verdict(stats: dict, arms: pd.DataFrame, stat: dict, tau_tbl: pd.DataFrame,
                     ref: dict, regime: dict) -> str:
    L = []
    fl = stats["flips"]
    n = stats["acc"]["n"]
    L.append(f"Flips: {fl['helped']} helped, {fl['harmed']} harmed, {fl['both_right']} both right, "
             f"{fl['both_wrong']} both wrong (n = {n}). Decision is live on "
             f"{100 * fl['decision_live_frac']:.0f}% of queries. "
             f"Oracle headroom over always-retrieve: {ref['headroom_over_always']:+.3f}.")

    sep_ok = bool(len(arms)) and bool(arms["separated"].any())
    stat_ok = bool(stat) and np.isfinite(stat.get("lo", np.nan)) and (stat["lo"] > 0 or stat["hi"] < 0)
    if stat_ok:
        L.append(f"Dissociation statistic: {stat['diff']:+.3f} [{stat['lo']:+.3f}, {stat['hi']:+.3f}] — "
                 "confidence and retrieval signals attach to different arms.")

    best = tau_tbl.iloc[0] if len(tau_tbl) else None
    single = tau_tbl[tau_tbl["n_features"] == 1]
    best_single = single["V@50%"].max() if len(single) else float("nan")
    beats_proxy = bool(best is not None and np.isfinite(best_single) and best["V@50%"] > best_single)
    if best is not None:
        L.append(f"Best feature set at a 50% budget: **{best['feature_set']}** "
                 f"(V = {best['V@50%']:.3f}); best single-feature proxy gate: {best_single:.3f}; "
                 f"always-retrieve: {ref['always_retrieve']:.3f}.")

    if stat_ok and beats_proxy:
        L.append("**Verdict: PROCEED.** The two signal families attach to different arms, and combining them "
                 "beats every single-proxy gate at matched budget. That is the argument for estimating tau "
                 "rather than thresholding a proxy.")
    elif stat_ok:
        L.append("**Verdict: PROCEED with care.** The structural dissociation holds, but the combined feature set "
                 "does not yet beat the best single proxy on policy value — likely an n problem. Scale up before "
                 "drawing conclusions about which gate wins.")
    elif fl["harmed"] >= max(20, 0.05 * n):
        L.append("**Verdict: PROCEED on harm.** No clean arm separation yet, but harm is frequent enough that "
                 "CATE estimation has something to fit.")
    else:
        L.append("**Verdict: RE-CHECK.** Neither arm separation nor substantial harm. Check the retriever tier "
                 "and the dataset before changing the model.")

    if regime:
        L.append(f"_Regime: {regime['regime']}. Which family wins is expected to flip between regimes; "
                 "running both a model-limited and a corpus-limited dataset is what makes that visible._")
    if n < 500:
        L.append(f"_n = {n}: indicative only. Cells with n < {MIN_CELL_N} are suppressed in the maps below._")
    return "\n\n".join(L)


# ---------------------------------------------------------------- plotting
def plot_arm_plane(arms: pd.DataFrame, path: str) -> None:
    """Each feature placed by what it knows about arm 0 vs arm 1."""
    if arms is None or arms.empty:
        return
    colors = {"F0-confidence": "#c0392b", "F0-popularity": "#e67e22",
              "F0-ambiguity": "#8e44ad", "F1-retrieval": "#2471a3"}
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    for _, r in arms.iterrows():
        c = colors.get(r["tier"], "grey")
        ax.errorbar(r["auc_Y0"], r["auc_Y1"],
                    xerr=[[max(0, r["auc_Y0"] - r["Y0_lo"])], [max(0, r["Y0_hi"] - r["auc_Y0"])]],
                    yerr=[[max(0, r["auc_Y1"] - r["Y1_lo"])], [max(0, r["Y1_hi"] - r["auc_Y1"])]],
                    fmt="o", color=c, ms=6, capsize=2, lw=1, alpha=0.85)
        ax.annotate(r["feature"], (r["auc_Y0"], r["auc_Y1"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    lim = (0.0, 1.0)
    ax.plot(lim, lim, color="lightgrey", lw=0.9, zorder=0)
    ax.axhline(0.5, color="grey", lw=0.7, ls="--"); ax.axvline(0.5, color="grey", lw=0.7, ls="--")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("AUC predicting arm 0 (parametric answer correct)")
    ax.set_ylabel("AUC predicting arm 1 (retrieval answer correct)")
    ax.set_title("Which arm does each signal know about?\nAbove the diagonal: knows the retrieval arm better")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in colors.items()],
              fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_tau_eval(tau_tbl: pd.DataFrame, ref: dict, path: str) -> None:
    if tau_tbl is None or tau_tbl.empty:
        return
    rate_cols = [c for c in tau_tbl.columns if c.startswith("V@") and c.endswith("%")]
    rates = [int(c[2:-1]) / 100 for c in rate_cols]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    for _, r in tau_tbl.iterrows():
        style = "--" if r["n_features"] == 1 else "-"
        ax.plot(rates, [r[c] for c in rate_cols], style, marker="o", ms=4, lw=1.4, label=str(r["feature_set"]))
    ax.axhline(ref["oracle"], color="black", ls=":", lw=1, label="oracle")
    ax.axhline(ref["always_retrieve"], color="grey", ls="-.", lw=1, label="always retrieve")
    ax.axhline(ref["never_retrieve"], color="grey", ls=":", lw=1, label="never retrieve")
    ax.set_xlabel("retrieval rate"); ax.set_ylabel("realized value")
    ax.set_title("policy value by feature set"); ax.legend(fontsize=6.5, ncol=1)
    ax = axes[1]
    d = tau_tbl[tau_tbl["corr_tau_delta"].notna()].sort_values("corr_tau_delta")
    y = np.arange(len(d))
    err = np.zeros((2, len(d)))
    has = d["corr_lo"].notna().to_numpy()
    err[0, has] = (d["corr_tau_delta"] - d["corr_lo"]).to_numpy()[has]
    err[1, has] = (d["corr_hi"] - d["corr_tau_delta"]).to_numpy()[has]
    ax.barh(y, d["corr_tau_delta"], xerr=err, color=["grey" if k == 1 else "steelblue" for k in d["n_features"]],
            capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(d["feature_set"], fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("corr(tau_hat, observed Δ)"); ax.set_title("effect-prediction quality")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_effect_map(tables: Dict[str, pd.DataFrame], xlabel: str, ylabel: str, title: str, path: str,
                    min_n: int = MIN_CELL_N) -> None:
    if tables["mean"].empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, key, cmap in zip(axes, ("mean", "harmful"), ("RdYlGn", "Reds")):
        m, n = tables[key].copy(), tables["n"]
        vals = m.to_numpy(dtype=float).copy()
        counts = n.to_numpy(dtype=float)
        vals[counts < min_n] = np.nan          # suppress noise cells
        if key == "mean":
            v = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
            im = ax.imshow(vals, cmap=cmap, vmin=-v, vmax=v, origin="lower", aspect="auto")
        else:
            v = max(0.05, float(np.nanmax(vals))) if np.isfinite(vals).any() else 0.05
            im = ax.imshow(vals, cmap=cmap, vmin=0, vmax=v, origin="lower", aspect="auto")
        ax.set_xticks(range(m.shape[1])); ax.set_xticklabels([str(c) for c in m.columns])
        ax.set_yticks(range(m.shape[0])); ax.set_yticklabels([str(r) for r in m.index])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                c = counts[i, j]
                txt = f"{vals[i, j]:.2f}\n(n={int(c)})" if np.isfinite(vals[i, j]) else (f"n={int(c)}" if np.isfinite(c) else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="grey" if not np.isfinite(vals[i, j]) else "black")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title("mean Δ (containment)" if key == "mean" else "fraction harmed")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{title}   (cells with n < {min_n} suppressed)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_marginals(ms: Dict[str, pd.DataFrame], path: str) -> None:
    ms = {k: v for k, v in ms.items() if len(v)}
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(5 * len(ms), 3.8), squeeze=False)
    for ax, (name, m) in zip(axes[0], ms.items()):
        by = m.columns[0]
        ax.errorbar(m[by], m["mean_delta"], yerr=1.96 * m["se"], fmt="o-", capsize=3, label="mean Δ ± 95% CI")
        ax.plot(m[by], m["help_rate"], "^--", color="seagreen", label="help rate")
        ax.plot(m[by], m["harm_rate"], "s--", color="crimson", label="harm rate")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(f"{name} bin (0 = lowest)"); ax.set_ylabel("Δ containment acc"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_gates(curves: Dict[str, pd.DataFrame], path: str) -> None:
    if not curves:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    for name, c in curves.items():
        ax.plot(c["retrieval_rate"], c["value"], "-", lw=1.4, label=f"{name} (AUC {auc_of_curve(c):.3f})")
    any_c = next(iter(curves.values()))
    ax.plot(any_c["retrieval_rate"], any_c["random_value"], "--", color="grey", lw=1, label="random gate")
    ax.axhline(any_c.attrs["v_oracle"], color="black", lw=1, ls=":", label="oracle")
    ax.set_xlabel("retrieval rate"); ax.set_ylabel("mean outcome − cost")
    ax.set_title("policy value vs retrieval rate"); ax.legend(fontsize=7)
    ax = axes[1]
    name, c = max(curves.items(), key=lambda kv: auc_of_curve(kv[1]))
    ax.stackplot(c["retrieval_rate"], c["harm_incurred"], c["benefit_forgone"],
                 labels=["harm incurred", "benefit forgone"], colors=["crimson", "steelblue"], alpha=0.7)
    ax.plot(c["retrieval_rate"], c["regret"], "k-", lw=1, label="regret")
    ax.set_xlabel("retrieval rate"); ax.set_ylabel("regret vs oracle")
    ax.set_title(f"regret decomposition — best gate: {name}"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_delta_hist(feat: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(feat["delta_f1"].dropna(), bins=41, color="steelblue")
    axes[0].set_title("Δ token-F1"); axes[0].set_xlabel("Δ F1")
    vc = feat["delta_acc"].dropna().value_counts().sort_index()
    axes[1].bar([str(int(k)) for k in vc.index], vc.values,
                color=["crimson" if k < 0 else ("grey" if k == 0 else "seagreen") for k in vc.index])
    axes[1].set_title("Δ containment accuracy (primary)"); axes[1].set_xlabel("Δ acc")
    for i, v in enumerate(vc.values):
        axes[1].text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------- report
def _fmt(v, floatfmt="{:.3f}") -> str:
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else floatfmt.format(v)
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    return str(v)


def md_table(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:.3f}") -> str:
    if df is None or len(df) == 0:
        return "_(empty)_"
    d = (df.reset_index() if index else df).astype(object)
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, floatfmt) for v in row) + " |")
    return "\n".join(lines)


def examples_table(feat: pd.DataFrame, n: int, helpful: bool) -> str:
    col = "delta_acc"
    d = feat[feat[col] > 0] if helpful else feat[feat[col] < 0]
    d = d.sort_values("delta_f1", ascending=not helpful).head(n)
    rows = [{
        "Δacc": int(r[col]), "Δf1": round(float(r["delta_f1"]), 2),
        "question": str(r["question"])[:70],
        "gold": "; ".join(map(str, list(r["answers"])[:2]))[:45],
        "arm0": str(r.get("answer0", ""))[:38], "arm1": str(r.get("answer1", ""))[:38],
        "subj_chars": int(r.get("subj_chars", 0)),
        "conf": round(float(r.get("probe_maxprob_mean", np.nan)), 2),
    } for _, r in d.iterrows()]
    return md_table(pd.DataFrame(rows), index=False)


def write_report(path: str, cfg_dict: dict, versions: dict, data_meta: dict, bin_meta: dict, stats: dict,
                 arms: pd.DataFrame, stat: dict, tau_tbl: pd.DataFrame, ref: dict, regime: dict,
                 maps: Dict[str, Dict[str, pd.DataFrame]],
                 marginals: Dict[str, pd.DataFrame], gate_tables: Dict[str, pd.DataFrame],
                 gate_attrs: Dict[str, dict], audit: pd.DataFrame, feat: pd.DataFrame) -> str:
    L: List[str] = ["# Phase 1 report — arm decomposition, tau evaluation, effect maps\n"]

    L.append("## Setup")
    L.append(f"- dataset: `{cfg_dict['dataset']}`, n = {stats['acc']['n']} (requested {cfg_dict['n_queries']}, seed {cfg_dict['seed']})")
    L.append(f"- model: `{cfg_dict['model_name']}` (4-bit: {cfg_dict['load_in_4bit']}); retriever: `{cfg_dict['retriever']}` / `{cfg_dict['index_name']}`, top_k = {cfg_dict['top_k']}")
    L.append(f"- decoding: greedy, max_new_tokens = {cfg_dict['max_new_tokens']}, prefix_k = {cfg_dict['prefix_k']}")
    L.append(f"- versions: {json.dumps(versions)}")
    L.append(f"- sampling: {json.dumps({k: v for k, v in data_meta.items() if k != 'per_bin_counts'})}\n")

    L.append("## 1. Headline — which arm does each signal know about?\n")
    L.append("The decision quantity is tau(x) = mu1(x) − mu0(x). Both arms are observed, so each feature "
             "is scored against each arm separately. `D = AUC(Y1) − AUC(Y0)`: negative means the feature is "
             "about what the model already knows, positive means it is about what the corpus provides.\n")
    L.append("_No target here is derived from `helped`. Both the full-sample and the decision-live versions of "
             "that target are contaminated — with binary outcomes, `helped` on the live subset is identically "
             "`1 − Y0`, i.e. the level in disguise._\n")
    L.append(md_table(arms[["feature", "tier", "auc_Y0", "Y0_lo", "Y0_hi", "auc_Y1", "Y1_lo", "Y1_hi",
                            "D", "D_lo", "D_hi", "separated"]] if len(arms) else arms, index=False))
    if stat:
        L.append(f"\n**Dissociation statistic** — `{stat['retr_feature']}` vs `{stat['conf_feature']}`, "
                 f"paired bootstrap on D(retrieval) − D(confidence):")
        L.append(f"- confidence: AUC(Y0) = {stat['conf_auc_Y0']:.3f}, AUC(Y1) = {stat['conf_auc_Y1']:.3f}, D = {stat['D_conf']:+.3f}")
        L.append(f"- retrieval:  AUC(Y0) = {stat['retr_auc_Y0']:.3f}, AUC(Y1) = {stat['retr_auc_Y1']:.3f}, D = {stat['D_retr']:+.3f}")
        L.append(f"- **difference = {stat['diff']:+.3f} [{stat['lo']:+.3f}, {stat['hi']:+.3f}], "
                 f"P(>0) = {stat['p_gt_0']:.3f}, n = {stat['n']}**\n")

    L.append("## 1b. Does it matter for the decision? (cross-fitted tau)\n")
    L.append(f"tau_hat = mu1_hat − mu0_hat, 5-fold cross-fitted per feature set, then used to rank queries for "
             f"retrieval. `V@r` is the realized value when the top r fraction is retrieved. "
             f"Reference points: never = {ref['never_retrieve']:.3f}, always = {ref['always_retrieve']:.3f}, "
             f"oracle = {ref['oracle']:.3f} (headroom over always = {ref['headroom_over_always']:+.3f}).\n")
    L.append(md_table(tau_tbl, index=False))
    L.append("\n_CIs resample rows only; they do not propagate estimation uncertainty in tau_hat, so treat "
             "differences between adjacent feature sets as suggestive at this n._\n")
    if regime:
        L.append(f"**Regime:** {regime['regime']} — Y0 = {regime['Y0_mean']:.3f}, Y1 = {regime['Y1_mean']:.3f}. "
                 f"At a 50% retrieval budget, confidence-only scores {regime['V50_confidence_only']:.3f} and "
                 f"retrieval-only {regime['V50_retrieval_only']:.3f}: **{regime['winner_at_50pct']} wins here**. "
                 f"tau = mu1 − mu0, so whichever arm carries more variance dominates the effect.\n")

    L.append("## 2. Decision gate")
    L.append(decision_verdict(stats, arms, stat, tau_tbl, ref, regime) + "\n")

    L.append("## 3. Overall effect of retrieval")
    rows = []
    for m in OUTCOMES:
        s = stats.get(m)
        if not s:
            continue
        row = {"outcome": m, "Y0": s["y0_mean"], "Y1": s["y1_mean"], "mean Δ": s["delta_mean"], "se": s["delta_se"]}
        row.update({f"harm@{t}": v for t, v in s["harm_rate"].items()})
        row.update({f"help@{t}": v for t, v in s["help_rate"].items()})
        rows.append(row)
    L.append(md_table(pd.DataFrame(rows), index=False))
    L.append("\nFlip table (containment accuracy — the honest count):")
    L.append(md_table(flip_table(feat)))
    fl = stats["flips"]
    L.append(f"\nDecision is live on {100 * fl['decision_live_frac']:.0f}% of queries; the rest are decided either way.\n")

    L.append("## 4. Effect maps")
    for key, tab in maps.items():
        L.append(f"### {key}")
        L.append("Mean Δ (containment):")
        L.append(md_table(tab["mean"]))
        L.append("\nCounts:")
        L.append(md_table(tab["n"], floatfmt="{:.0f}") + "\n")

    L.append("## 5. Marginals")
    for name, m in marginals.items():
        if len(m):
            L.append(f"By {name}:")
            L.append(md_table(m, index=False) + "\n")

    L.append("## 6. Proxy-gate regret decomposition (cost = 0)")
    for name, tbl in gate_tables.items():
        a = gate_attrs[name]
        L.append(f"**{name}** — oracle {a['v_oracle']:.3f}, always {a['v_always']:.3f}, never {a['v_never']:.3f}, "
                 f"curve AUC {a.get('curve_auc', float('nan')):.3f}, n = {a['n']}")
        L.append(md_table(tbl[["retrieval_rate", "value", "random_value", "regret", "harm_incurred", "benefit_forgone"]],
                          index=False) + "\n")

    L.append("## 7. Harm audit — is negative Δ F1 real, or a scoring artifact?")
    if len(audit):
        L.append(md_table(audit, index=False))
        vc = audit["kind"].value_counts()
        L.append("\n" + md_table(vc.rename("count").to_frame(), floatfmt="{:.0f}"))
    else:
        L.append("_No negative-Δ F1 cases._")
    L.append("")

    L.append("## 8. Examples")
    L.append("Harmed (arm0 right → arm1 wrong):")
    L.append(examples_table(feat, 12, helpful=False))
    L.append("\nHelped:")
    L.append(examples_table(feat, 8, helpful=True) + "\n")

    text = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ------------------------------------------------------------------ driver
def run_analysis(feat: pd.DataFrame, cfg_dict: dict, versions: dict, data_meta: dict,
                 results_dir: str, figures_dir: str, thresholds=(0.25, 0.5, 0.75),
                 conf_bins: int = 5, pop_bins: int = 5, n_boot: int = 2000, **_ignored) -> dict:
    os.makedirs(results_dir, exist_ok=True); os.makedirs(figures_dir, exist_ok=True)
    conf_col = "probe_maxprob_mean"
    feat, bin_meta = add_bins(feat, conf_col, conf_bins, pop_bins)
    feat.to_parquet(os.path.join(results_dir, "features_binned.parquet"), index=False)

    stats = summary_stats(feat, thresholds)
    arms = arm_decomposition(feat, n_boot=n_boot)
    stat = dissociation_statistic(feat, n_boot=n_boot)
    tau_tbl, ref = tau_evaluation(feat, n_boot=min(n_boot, 400))
    regime = regime_diagnostic(feat, arms, tau_tbl)
    if len(arms):
        arms.to_csv(os.path.join(results_dir, "arm_decomposition.csv"), index=False)
    if len(tau_tbl):
        tau_tbl.to_csv(os.path.join(results_dir, "tau_evaluation.csv"), index=False)

    maps = {
        "retrieval score × ambiguity (the informative axes)": cell_tables(feat, "ambig_bin", "retr_bin"),
        "confidence × popularity (the literature's axes — contrast)": cell_tables(feat, "pop_bin", "conf_bin"),
    }
    marginals = {n: marginal(feat, c) for n, c in
                 (("popularity", "pop_bin"), ("confidence", "conf_bin"),
                  ("retrieval score", "retr_bin"), ("subject length (ambiguity)", "ambig_bin"))}
    audit = harm_audit(feat)
    if len(audit):
        audit.to_csv(os.path.join(results_dir, "harm_audit.csv"), index=False)

    gate_specs = [("uncertainty gate: prefix entropy", "probe_entropy_mean", True),
                  ("uncertainty gate: 1 − prefix max-prob", "probe_maxprob_mean", False),
                  ("popularity gate: −log_pop", "log_pop", False),
                  ("retrieval gate: normalized top-1", "retr_top1_z_bylen", True),
                  ("ambiguity gate: subject length", "subj_chars", True)]
    curves, gate_tables, gate_attrs = {}, {}, {}
    for name, col, higher in gate_specs:
        if col in feat.columns and feat[col].notna().sum() > 10 and feat[col].nunique() > 2:
            c = gate_curve(feat, col, "y0_acc", "y1_acc", higher_means_retrieve=higher, cost=0.0)
            c.attrs["curve_auc"] = auc_of_curve(c)
            curves[name] = c
            gate_tables[name] = curve_at_rates(c)
            gate_attrs[name] = dict(c.attrs)
            c.to_csv(os.path.join(results_dir, f"gate_curve_{len(curves)}.csv"), index=False)

    plot_arm_plane(arms, os.path.join(figures_dir, "arm_plane.png"))
    plot_tau_eval(tau_tbl, ref, os.path.join(figures_dir, "tau_evaluation.png"))
    plot_effect_map(maps["retrieval score × ambiguity (the informative axes)"],
                    "retrieval-score bin (0 = lowest)", "subject-length bin (0 = shortest/most ambiguous)",
                    "Effect map — informative axes", os.path.join(figures_dir, "effect_map_informative.png"))
    plot_effect_map(maps["confidence × popularity (the literature's axes — contrast)"],
                    "confidence bin (0 = least confident)", "popularity bin (0 = least popular)",
                    "Effect map — the axes gates use", os.path.join(figures_dir, "effect_map_contrast.png"))
    plot_marginals(marginals, os.path.join(figures_dir, "marginals.png"))
    plot_gates(curves, os.path.join(figures_dir, "gate_regret.png"))
    plot_delta_hist(feat, os.path.join(figures_dir, "delta_hist.png"))

    summary = {"stats": stats, "dissociation_statistic": stat, "regime": regime, "reference": ref,
               "arm_decomposition": arms.to_dict("records") if len(arms) else [],
               "tau_evaluation": tau_tbl.to_dict("records") if len(tau_tbl) else [],
               "bins": bin_meta, "data_meta": data_meta, "versions": versions, "config": cfg_dict,
               "maps": {k: {kk: vv.to_dict() for kk, vv in t.items() if len(vv)} for k, t in maps.items()},
               "gate_attrs": gate_attrs,
               "harm_audit_kinds": audit["kind"].value_counts().to_dict() if len(audit) else {}}
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    report = write_report(os.path.join(results_dir, "report.md"), cfg_dict, versions, data_meta, bin_meta, stats,
                          arms, stat, tau_tbl, ref, regime, maps, marginals, gate_tables, gate_attrs, audit, feat)
    return {"summary": summary, "report": report, "feat": feat,
            "arms": arms, "tau": tau_tbl, "statistic": stat, "regime": regime, "reference": ref}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
