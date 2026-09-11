# Retrieval as Treatment

Causal effect estimation and off-policy evaluation for adaptive retrieval-augmented generation.
This repository is the reproducibility package for the research plan in `docs/` (see the plan for the full method).

**Status:** Phase 0 (setup) + Phase 1 (harm map) implemented. Phases 2–5 (full factorial, CATE estimators, OPE tool, multi-arm) come next.

## What Phase 1 does

For a popularity-stratified sample of PopQA questions it runs two arms with an identical prompt:

| arm | treatment | what is logged |
|---|---|---|
| 0 | parametric answer, no passages | answer, token-F1 / EM / containment-acc, **first-k-token confidence probe** (max-prob, entropy), answer log-prob, "I don't know" flag |
| 1 | same prompt + top-k BM25 Wikipedia passages | answer, scores, #passages used |

and then produces:

- **Harm map** — fraction of queries where retrieval flips a right answer to wrong, as a function of (arm-0 confidence × entity popularity).
- **Proxy-gate regret decomposition** — for an uncertainty gate swept over thresholds, how much value is lost by *retrieving where it hurts* vs *not retrieving where it helps*, relative to the oracle that observes both arms.
- A `report.md` you can paste back, plus `summary.json`, per-query features/outcomes (parquet), and PNG figures.

## Run it on Colab (recommended)

1. Push this repo to GitHub.
2. Open `notebooks/01_harm_map.ipynb` in Colab (File → Open notebook → GitHub, or upload it). Runtime → **GPU**.
3. In cell 1 set `REPO_URL` to your repo and leave `WORKDIR` as the Drive path. Run all cells.
4. Start with `N_QUERIES = 100` as a smoke test (~10 min on a T4 after downloads), then set `2000` and re-run — the 100 already done are reused.

Everything generated goes to Google Drive under `WORKDIR`, one JSON line per result as it completes. If Colab disconnects, re-run from the top: each step skips what is already done.

Compute (Qwen2.5-7B-Instruct, 4-bit, 2,000 queries × 2 arms):
- T4: ~1 h of generation; L4/A100: ~15–20 min.
- One-time downloads: model ~15 GB (or ~5.5 GB using `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` — swap `model_name` in `configs/phase1_popqa.yaml`), BM25 index ~2.5 GB (persisted to Drive after the first download).

CLI alternative (same steps, no notebook):
```bash
python scripts/run_phase1.py --workdir /content/drive/MyDrive/retrieval-as-treatment --n 2000
python scripts/run_phase1.py --workdir ... --skip-generation   # rebuild features + report from cached logs
```

## What to send back after the run

- `results/phase1/report.md` (printed at the end of the notebook)
- `figures/phase1/harm_map_f1.png` and `harm_map_acc.png`
- optionally `results/phase1/summary.json`

The report contains a **decision gate** verdict: PROCEED if retrieval harm is non-trivial (≥5% overall or ≥10% in any harm-map cell); otherwise re-check with a smaller LLM before investing in Phase 2.

## Layout

```
rat/                 package (installed with `pip install -e .`)
  config.py          Config dataclass, YAML loading, workdir layout
  data.py            PopQA / NQ-open / TriviaQA loaders, popularity-stratified sampling
  retrieve.py        BM25 via Pyserini prebuilt Wikipedia index (Drive-persisted), cached-JSONL fallback
  generate.py        4-bit LLM, both-arm prompts, first-k-token confidence probe, answer cleaning
  score.py           token-F1 / EM / containment accuracy
  logs.py            resumable JSONL store (append-as-you-go, skip done, tolerate torn writes)
  features.py        F0 / F1 feature table joined with outcomes
  analysis.py        harm map, marginals, gate regret decomposition, figures, report
  phase1.py          step_data → step_retrieve → step_generate(0/1) → step_features → step_analysis
configs/phase1_popqa.yaml
notebooks/01_harm_map.ipynb
scripts/run_phase1.py
tests/test_offline.py   # python tests/test_offline.py  (no GPU / network needed)
```

## Workdir layout (Drive)

```
WORKDIR/
  data/            sampled query table (parquet) + sampling meta (bin edges)
  retrievals/      popqa__bm25__k10.jsonl
  logs/gen/        popqa__<model>__arm0.jsonl, popqa__<model>__bm25_top5__arm1.jsonl
  features/        joined feature/outcome table (parquet)
  results/phase1/  report.md, summary.json, gate_curve_*.csv, features_binned.parquet, versions.json
  figures/phase1/  harm_map_f1.png, harm_map_acc.png, marginals.png, gate_regret_*.png, delta_hist.png
  pyserini_cache/  persisted BM25 index
```

## Definitions used in the report

- `Δ = Y(1) − Y(0)` per query; **harmful** if `Δ < −0.5`, **helpful** if `Δ > 0.5` (sensitivity at 0.25 / 0.75 also reported).
- Primary `Y` is token-level F1 (max over gold aliases); `acc` is PopQA's containment metric.
- Gate regret = oracle value − gate value = harm incurred + benefit forgone (exact identity, tested).

## Troubleshooting

- **`java not found` / version < 21** → re-run the deps cell; Pyserini needs Java 21.
- **Prebuilt index name error** → the error message lists available Wikipedia indexes; set `index_name` in the config.
- **CUDA OOM** → lower `batch_size` (8 → 4) or `max_prompt_tokens`.
- **HF dataset load fails** → dataset names drift; see `rat/data.py` (`akariasai/PopQA`, `google-research-datasets/nq_open`, `mandarjoshi/trivia_qa`).
- **torch got downgraded by pip** → Runtime → Restart, re-run cell 3 onward (do not re-run the pip cell).
