"""Phase 1 analysis: the harm map and the proxy-gate regret decomposition.

Definitions (research plan §1.3, §4.4):
  delta_i         = Y_i^(1) - Y_i^(0)
  harmful         : delta_i < -thr          helpful: delta_i > thr
  proxy gate      : retrieve iff score(q) >= t   (score = an uncertainty proxy)
  oracle (cost c) : retrieve iff delta_i > c
  regret(gate)    = V(oracle) - V(gate) = harm_incurred + benefit_forgone, where
     harm_incurred   = mean( 1[retrieved]     * max(0, c - delta) )
     benefit_forgone = mean( 1[not retrieved] * max(0, delta - c) )
This identity is exact (checked numerically in tests).
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

OUTCOMES = ("f1", "em", "acc")


# ----------------------------------------------------------------- binning
def quantile_bins(x: pd.Series, n: int) -> Tuple[pd.Series, np.ndarray]:
    codes, edges = pd.qcut(x, q=n, labels=False, retbins=True, duplicates="drop")
    return codes, edges


def add_bins(feat: pd.DataFrame, conf_col: str, n_conf: int, n_pop: int) -> Tuple[pd.DataFrame, dict]:
    """Adds conf_bin (quantiles of the confidence proxy) and pop_bin (kept from
    sampling if present, else quantiles of log_pop)."""
    f = feat.copy()
    meta: dict = {"conf_col": conf_col}

    f["conf_bin"] = -1
    ok = f[conf_col].notna()
    if ok.sum() >= n_conf:
        codes, edges = quantile_bins(f.loc[ok, conf_col], n_conf)
        f.loc[ok, "conf_bin"] = codes.astype(int).values
        meta["conf_bin_edges"] = [float(e) for e in edges]

    have_pop = "pop_bin" in f.columns and (f["pop_bin"] >= 0).any()
    if not have_pop and "log_pop" in f.columns and f["log_pop"].notna().sum() >= n_pop:
        codes, edges = quantile_bins(f["log_pop"], n_pop)
        f["pop_bin"] = codes.astype(int)
        meta["pop_bin_edges_log10"] = [float(e) for e in edges]
    elif have_pop and "log_pop" in f.columns:
        meta["pop_bin_ranges_log10"] = (
            f.groupby("pop_bin")["log_pop"].agg(["min", "max"]).round(2).to_dict("index")
        )
    if "pop_bin" not in f.columns:
        f["pop_bin"] = -1
    return f, meta


# ------------------------------------------------------------ cell tables
def cell_tables(feat: pd.DataFrame, delta_col: str, thr: float, row: str = "pop_bin", col: str = "conf_bin") -> Dict[str, pd.DataFrame]:
    d = feat.dropna(subset=[delta_col])
    d = d[(d[row] >= 0) & (d[col] >= 0)]
    g = d.groupby([row, col])[delta_col]
    return {
        "harmful": g.apply(lambda s: float((s < -thr).mean())).unstack(col),
        "helpful": g.apply(lambda s: float((s > thr).mean())).unstack(col),
        "mean": g.mean().unstack(col),
        "n": g.size().unstack(col),
    }


def marginal(feat: pd.DataFrame, by: str, delta_col: str, thr: float) -> pd.DataFrame:
    d = feat.dropna(subset=[delta_col])
    d = d[d[by] >= 0]
    g = d.groupby(by)[delta_col]
    size = g.size()
    out = pd.DataFrame(
        {
            "n": size,
            "mean_delta": g.mean(),
            "se": g.std(ddof=1) / np.sqrt(size.clip(lower=1)),
            "harm_rate": g.apply(lambda s: float((s < -thr).mean())),
            "help_rate": g.apply(lambda s: float((s > thr).mean())),
        }
    )
    return out.reset_index()


# ------------------------------------------------------------- gate curve
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
        v_gate = float(np.where(R, y1 - cost, y0).mean())
        harm = float((R * np.clip(cost - delta, 0, None)).mean())
        forgone = float(((~R) * np.clip(delta - cost, 0, None)).mean())
        rate = float(R.mean())
        rows.append(
            {
                "threshold": float(t),
                "retrieval_rate": rate,
                "value": v_gate,
                "random_value": rate * v_always + (1 - rate) * v_never,
                "oracle_value": v_oracle,
                "regret": v_oracle - v_gate,
                "harm_incurred": harm,
                "benefit_forgone": forgone,
            }
        )
    out = pd.DataFrame(rows).sort_values("retrieval_rate").reset_index(drop=True)
    out.attrs.update({"v_oracle": v_oracle, "v_always": v_always, "v_never": v_never, "n": int(len(d)), "cost": cost})
    return out


def curve_at_rates(curve: pd.DataFrame, rates=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> pd.DataFrame:
    idx = [int((curve["retrieval_rate"] - r).abs().idxmin()) for r in rates]
    return curve.loc[idx].reset_index(drop=True)


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
    return s


# ---------------------------------------------------------------- plotting
def plot_harm_map(tables: Dict[str, pd.DataFrame], xlabel: str, ylabel: str, title: str, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, cmap in zip(axes, ("harmful", "helpful"), ("Reds", "Greens")):
        m, n = tables[key], tables["n"]
        vals = m.to_numpy(dtype=float)
        vmax = max(0.05, float(np.nanmax(vals))) if np.isfinite(vals).any() else 0.05
        im = ax.imshow(vals, cmap=cmap, vmin=0, vmax=vmax, origin="lower", aspect="auto")
        ax.set_xticks(range(m.shape[1])); ax.set_xticklabels([str(c) for c in m.columns])
        ax.set_yticks(range(m.shape[0])); ax.set_yticklabels([str(r) for r in m.index])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = vals[i, j]
                if np.isfinite(v):
                    c = n.to_numpy()[i, j]
                    ax.text(j, i, f"{v:.2f}\n(n={int(c)})", ha="center", va="center", fontsize=8)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(f"fraction {key}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_marginals(m_pop: pd.DataFrame, m_conf: pd.DataFrame, path: str, delta_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m, by in zip(axes, (m_pop, m_conf), ("pop_bin", "conf_bin")):
        if m.empty:
            ax.set_visible(False)
            continue
        ax.errorbar(m[by], m["mean_delta"], yerr=1.96 * m["se"], fmt="o-", capsize=3, label="mean Δ ± 95% CI")
        ax.plot(m[by], m["harm_rate"], "s--", color="crimson", label="harm rate")
        ax.plot(m[by], m["help_rate"], "^--", color="seagreen", label="help rate")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(f"{by} (0 = lowest)"); ax.set_ylabel(delta_label); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_gate(curve: pd.DataFrame, path: str, score_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(curve["retrieval_rate"], curve["value"], "o-", ms=3, label=f"gate on {score_label}")
    ax.plot(curve["retrieval_rate"], curve["random_value"], "--", color="grey", label="random gate")
    ax.axhline(curve.attrs.get("v_oracle", np.nan), color="black", lw=1, ls=":", label="oracle (both arms observed)")
    ax.set_xlabel("retrieval rate"); ax.set_ylabel("mean outcome − cost"); ax.legend(fontsize=8)
    ax.set_title("policy value vs retrieval rate")
    ax = axes[1]
    ax.stackplot(curve["retrieval_rate"], curve["harm_incurred"], curve["benefit_forgone"],
                 labels=["harm incurred (retrieved where Δ<c)", "benefit forgone (skipped where Δ>c)"],
                 colors=["crimson", "steelblue"], alpha=0.7)
    ax.plot(curve["retrieval_rate"], curve["regret"], "k-", lw=1, label="regret = sum")
    ax.set_xlabel("retrieval rate"); ax.set_ylabel("regret vs oracle"); ax.legend(fontsize=8)
    ax.set_title("proxy-gate regret decomposition")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_delta_hist(feat: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(feat["delta_f1"].dropna(), bins=41, color="steelblue")
    axes[0].set_title("Δ token-F1 (arm1 − arm0)"); axes[0].set_xlabel("Δ F1")
    vc = feat["delta_acc"].dropna().value_counts().sort_index()
    axes[1].bar([str(int(k)) for k in vc.index], vc.values, color=["crimson" if k < 0 else ("grey" if k == 0 else "seagreen") for k in vc.index])
    axes[1].set_title("Δ containment-accuracy"); axes[1].set_xlabel("Δ acc (−1 harmful, +1 helpful)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------- report
def _fmt(v, floatfmt="{:.3f}") -> str:
    if isinstance(v, float):
        return "" if np.isnan(v) else floatfmt.format(v)
    return str(v)


def md_table(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:.3f}") -> str:
    d = (df.reset_index() if index else df).astype(object)  # keep ints as ints
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, floatfmt) for v in row) + " |")
    return "\n".join(lines)


def examples_table(feat: pd.DataFrame, n: int, harmful: bool) -> str:
    d = feat.dropna(subset=["delta_f1"])
    if "probe_maxprob_mean" in d.columns:  # among ties, show the confident-and-hurt (or unsure-and-helped) cases first
        d = d.sort_values(["delta_f1", "probe_maxprob_mean"], ascending=[harmful, not harmful])
    else:
        d = d.sort_values("delta_f1", ascending=harmful)
    d = d.head(n)
    rows = []
    for _, r in d.iterrows():
        golds = list(r["answers"])[:3] if "answers" in d.columns and r["answers"] is not None else []
        rows.append(
            {
                "Δf1": _fmt(float(r["delta_f1"])),
                "question": str(r["question"])[:90],
                "gold": "; ".join(map(str, golds))[:60],
                "arm0 answer": str(r.get("answer0", ""))[:50],
                "arm1 answer": str(r.get("answer1", ""))[:50],
                "log_pop": _fmt(float(r["log_pop"])) if "log_pop" in d.columns else "",
                "probe_maxprob": _fmt(float(r.get("probe_maxprob_mean", np.nan))),
            }
        )
    return md_table(pd.DataFrame(rows), index=False)


def decision_verdict(stats: dict, tables_f1: Dict[str, pd.DataFrame], thr: float) -> str:
    overall = stats["f1"]["harm_rate"][str(thr)]
    harm_cells = tables_f1["harmful"].to_numpy(dtype=float)
    max_cell = float(np.nanmax(harm_cells)) if np.isfinite(harm_cells).any() else float("nan")
    lines = [f"Overall harm rate (Δ F1 < −{thr}): **{overall:.3f}**; max harm-map cell: **{max_cell:.3f}**."]
    if overall >= 0.05 or max_cell >= 0.10:
        lines.append("**Verdict: PROCEED.** Retrieval harm is non-trivial — the treatment-effect framing has something to explain.")
    elif overall >= 0.02:
        lines.append("**Verdict: MARGINAL.** Harm exists but is small; check the dense retriever tier and/or a smaller LLM before committing.")
    else:
        lines.append("**Verdict: RE-CHECK.** Harm is negligible for this (model, retriever). Switch to a smaller LLM (weaker parametric knowledge → more heterogeneous effects) and re-run Phase 1.")
    return "\n".join(lines)


def write_report(path: str, cfg_dict: dict, versions: dict, data_meta: dict, bin_meta: dict, stats: dict,
                 tables: Dict[str, Dict[str, pd.DataFrame]], m_pop: pd.DataFrame, m_conf: pd.DataFrame,
                 gate_tables: Dict[str, pd.DataFrame], gate_attrs: Dict[str, dict], feat: pd.DataFrame,
                 harm_thr: float) -> str:
    L: List[str] = []
    L.append("# Phase 1 report — harm map\n")
    L.append("## Setup")
    L.append(f"- dataset: `{cfg_dict['dataset']}`, n = {stats['f1']['n']} (requested {cfg_dict['n_queries']}, seed {cfg_dict['seed']})")
    L.append(f"- model: `{cfg_dict['model_name']}` (4-bit: {cfg_dict['load_in_4bit']}); retriever: `{cfg_dict['retriever']}` / `{cfg_dict['index_name']}`, top_k = {cfg_dict['top_k']}")
    L.append(f"- decoding: greedy, max_new_tokens = {cfg_dict['max_new_tokens']}, prefix_k = {cfg_dict['prefix_k']}")
    L.append(f"- versions: {json.dumps(versions)}")
    L.append(f"- sampling: {json.dumps({k: v for k, v in data_meta.items() if k != 'per_bin_counts'})}")
    L.append(f"- bins: {json.dumps(bin_meta)}\n")

    L.append("## Overall effect of retrieval")
    rows = []
    for m in OUTCOMES:
        if m not in stats:
            continue
        s = stats[m]
        row = {"outcome": m, "Y0 (no retrieval)": s["y0_mean"], "Y1 (retrieval)": s["y1_mean"],
               "mean Δ": s["delta_mean"], "se": s["delta_se"]}
        for t, v in s["harm_rate"].items():
            row[f"harm@{t}"] = v
        for t, v in s["help_rate"].items():
            row[f"help@{t}"] = v
        rows.append(row)
    L.append(md_table(pd.DataFrame(rows), index=False) + "\n")

    L.append("## Decision gate")
    L.append(decision_verdict(stats, tables["f1"], harm_thr) + "\n")

    for m in ("f1", "acc"):
        if m not in tables:
            continue
        L.append(f"## Harm map — Δ {m} (rows: pop_bin, 0 = least popular; cols: conf_bin, 0 = least confident)")
        L.append(f"Fraction harmful (Δ < −{harm_thr}):")
        L.append(md_table(tables[m]["harmful"]))
        L.append(f"\nFraction helpful (Δ > {harm_thr}):")
        L.append(md_table(tables[m]["helpful"]))
        L.append("\nMean Δ:")
        L.append(md_table(tables[m]["mean"]))
        L.append("\nCounts:")
        L.append(md_table(tables[m]["n"], floatfmt="{:.0f}") + "\n")

    L.append("## Marginals (Δ F1)")
    L.append("By popularity bin:")
    L.append(md_table(m_pop, index=False))
    L.append("\nBy confidence bin (arm-0 prefix max-prob):")
    L.append(md_table(m_conf, index=False) + "\n")

    L.append("## Proxy-gate regret decomposition (cost = 0; gate retrieves when the proxy says 'uncertain')")
    for name, tbl in gate_tables.items():
        a = gate_attrs[name]
        L.append(f"**{name}** — oracle value {a['v_oracle']:.3f}, always-retrieve {a['v_always']:.3f}, never-retrieve {a['v_never']:.3f}, n = {a['n']}")
        L.append(md_table(tbl[["retrieval_rate", "value", "random_value", "regret", "harm_incurred", "benefit_forgone"]], index=False) + "\n")

    L.append("## Examples — most harmful (arm0 right, arm1 wrong)")
    L.append(examples_table(feat, 10, harmful=True) + "\n")
    L.append("## Examples — most helpful")
    L.append(examples_table(feat, 5, harmful=False) + "\n")

    text = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ------------------------------------------------------------------ driver
def run_analysis(feat: pd.DataFrame, cfg_dict: dict, versions: dict, data_meta: dict,
                 results_dir: str, figures_dir: str, thresholds=(0.25, 0.5, 0.75), harm_thr: float = 0.5,
                 conf_bins: int = 5, pop_bins: int = 5) -> dict:
    os.makedirs(results_dir, exist_ok=True); os.makedirs(figures_dir, exist_ok=True)
    conf_col = "probe_maxprob_mean"
    feat, bin_meta = add_bins(feat, conf_col, conf_bins, pop_bins)
    feat.to_parquet(os.path.join(results_dir, "features_binned.parquet"), index=False)

    stats = summary_stats(feat, thresholds)
    tables = {m: cell_tables(feat, f"delta_{m}", harm_thr) for m in ("f1", "acc") if f"delta_{m}" in feat.columns}
    m_pop = marginal(feat, "pop_bin", "delta_f1", harm_thr)
    m_conf = marginal(feat, "conf_bin", "delta_f1", harm_thr)

    gate_tables, gate_attrs, curves = {}, {}, {}
    for name, col, higher in (("uncertainty gate: prefix entropy", "probe_entropy_mean", True),
                              ("uncertainty gate: 1 − prefix max-prob", "probe_maxprob_mean", False)):
        if col in feat.columns and feat[col].notna().sum() > 10:
            c = gate_curve(feat, col, "y0_f1", "y1_f1", higher_means_retrieve=higher, cost=0.0)
            curves[name] = c
            gate_tables[name] = curve_at_rates(c)
            gate_attrs[name] = dict(c.attrs)
    if "log_pop" in feat.columns and feat["log_pop"].notna().sum() > 10:
        c = gate_curve(feat, "log_pop", "y0_f1", "y1_f1", higher_means_retrieve=False, cost=0.0)
        curves["popularity gate: −log_pop"] = c
        gate_tables["popularity gate: −log_pop"] = curve_at_rates(c)
        gate_attrs["popularity gate: −log_pop"] = dict(c.attrs)

    # figures
    for m in tables:
        plot_harm_map(tables[m], "conf_bin (arm-0 prefix max-prob, 0 = least confident)",
                      "pop_bin (0 = least popular)", f"Harm map — Δ {m}, threshold {harm_thr}",
                      os.path.join(figures_dir, f"harm_map_{m}.png"))
    plot_marginals(m_pop, m_conf, os.path.join(figures_dir, "marginals.png"), "Δ F1")
    plot_delta_hist(feat, os.path.join(figures_dir, "delta_hist.png"))
    for i, (name, c) in enumerate(curves.items()):
        plot_gate(c, os.path.join(figures_dir, f"gate_regret_{i}.png"), name)
        c.to_csv(os.path.join(results_dir, f"gate_curve_{i}.csv"), index=False)

    summary = {"stats": stats, "bins": bin_meta, "data_meta": data_meta, "versions": versions, "config": cfg_dict,
               "harm_tables": {m: {k: v.to_dict() for k, v in t.items()} for m, t in tables.items()},
               "gate_attrs": gate_attrs}
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    report = write_report(os.path.join(results_dir, "report.md"), cfg_dict, versions, data_meta, bin_meta, stats,
                          tables, m_pop, m_conf, gate_tables, gate_attrs, feat, harm_thr)
    return {"summary": summary, "report": report, "feat": feat}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)
