# Retrieval as Treatment

Causal effect estimation and off-policy evaluation for adaptive retrieval-augmented generation.
Reproducibility package for the plan in `docs/RESEARCH_PLAN.md`.

**Status:** Phase 0 (setup) + Phase 1 (dissociation & effect maps) implemented, revised after an n=100 pilot. Phases 2–5 (full factorial, CATE estimators, OPE tool, multi-arm) next.

## The claim Phase 1 tests

The decision quantity is `tau(x) = mu1(x) - mu0(x)`: the change in correctness caused by retrieving. Phase 1 asks two questions, both answerable because **both arms are observed for every query**:

1. **Structural.** Which arm does each pre-decision signal know about? Confidence should attach to the parametric arm, retrieval scores to the retrieval arm. Neither alone determines `tau`.
2. **Decision-relevant.** Does a cross-fitted `tau_hat` from each feature set actually drive a better retrieval decision at matched budget than the single-proxy gates in the literature?

> **The regime finding (n=300 × 2):** which family of signals wins *flips by dataset*. On PopQA the model is at a floor (Y⁰=0.17), so `mu0` barely varies, `tau ≈ mu1`, and retrieval signals win. On TriviaQA the model is competent (Y⁰=0.56), `mu0` varies, and confidence wins. No fixed proxy is correct across regimes — which is the argument for estimating `tau` rather than thresholding a proxy.

## Two circular targets, and why neither is used

Documented because both were implemented and both had to be thrown out:

- `AUC(feature -> helped)` over all queries: `helped` requires arm 0 to be wrong, so any predictor of the level anti-predicts it for free.
- `AUC(feature -> helped)` on the **decision-live** subset: worse. With binary outcomes, "the arms disagree" means exactly one is right, so on that subset `helped` is *identically* `1 - Y0`. It is the level wearing a disguise. Verified on both datasets.

Everything reported is measured against `Y0`, `Y1`, or `Delta` directly. `tests/test_offline.py::test_no_circular_target_used` fails if `helped` reappears as a target.

Separation uses `|AUC - 0.5|`, not raw AUC: features like entropy point downward by construction, and a raw difference would misread them as knowing the retrieval arm.

## What changed after the pilot (n=100, PopQA, Qwen2.5-7B)

| pilot finding | change |
|---|---|
| Confidence tracks the parametric arm; retrieval scores track the retrieval arm. | **Arm decomposition** + **cross-fitted tau evaluation** are the headline analyses (`rat/dissociation.py`), with bootstrap CIs and a paired dissociation statistic. The harm map is demoted to support. |
| Only 3/100 genuine harm cases; both designed map axes flat. | Effect maps now use **retrieval score × ambiguity** as the informative axes, keeping **confidence × popularity** as an explicit contrast figure. Cells with n < 20 are suppressed. |
| All 3 harm cases were short polysemous names (`Idaho`, `Summit`, `X`). | Added **ambiguity features**: subject length, token count, single-token flag, plus corpus-side title-match features. |
| Raw BM25 top-1 correlates r≈0.70 with subject-name length; residualised AUC falls 0.906 → 0.663. | Added **normalized retrieval features** (rank ratios, CV, z-within-query-length, residualised score). Raw score kept for reference only. |
| 4 of 7 negative-ΔF1 cases were scoring artifacts, not flips. | **Containment accuracy** is the primary effect definition; a **harm audit** table labels every negative case genuine vs artifact. |
| 63% of queries wrong in both arms; decision live on 20%. | **TriviaQA added** alongside PopQA, whose long tail is where retrieval nearly always helps. Reports show the decision-live fraction. |
| Verdict fired on a cell with n=3. | Verdict now keys on the dissociation, with a minimum-cell rule. |

**One methodological note.** `helped` requires arm 0 to be wrong, so any predictor of the level mechanically anti-predicts `helped`. The primary column `auc_effect_live` is therefore computed on the decision-live subset (arms disagree), which removes that coupling. Full-sample AUCs are still written to `dissociation.csv`.

## Run it on Colab

1. Push this repo to GitHub.
2. Open `notebooks/01_dissociation.ipynb` in Colab. Runtime → **GPU**.
3. Set `REPO_URL` in cell 1. Run all cells.
4. Start at `N_POPQA = 300`, `N_TRIVIAQA = 300` (~35 min on a T4 after downloads). If the verdict is PROCEED, raise to 2000/1500 and re-run — nothing already computed is repeated.

Everything is written to Drive one JSON line at a time. A disconnect costs nothing: re-run from the top.

Compute (Qwen2.5-7B-Instruct, 4-bit, per 1,000 queries × 2 arms): T4 ≈ 30 min, L4/A100 ≈ 8 min. One-time downloads: model ~15 GB (or ~5.5 GB with `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` — swap `model_name` in `configs/phase1.yaml`), BM25 index ~2.5 GB (persisted to Drive).

CLI alternative:
```bash
python scripts/run_phase1.py --workdir /content/drive/MyDrive/retrieval-as-treatment \
    --datasets popqa triviaqa --n-popqa 2000 --n-triviaqa 1500
python scripts/run_phase1.py --workdir ... --skip-generation    # rebuild reports from cached logs, no GPU
```

## What to send back

1. `results/phase1_combined/summary.md` — cross-dataset table
2. `results/phase1/<dataset>/report.md` (both datasets)
3. `results/phase1/<dataset>/features_binned.parquet` (both)
4. `figures/phase1/<dataset>/arm_plane.png` and `tau_evaluation.png`

## Layout

```
rat/
  config.py        Config dataclass, YAML loading, workdir layout
  data.py          PopQA / TriviaQA / NQ-open loaders, popularity-stratified sampling
  retrieve.py      BM25 via Pyserini prebuilt index (Drive-persisted), cached-JSONL fallback
  generate.py      4-bit LLM, both-arm prompts, first-k-token confidence probe
  score.py         token-F1 / EM / containment accuracy
  logs.py          resumable JSONL store
  features.py      F0 (confidence, popularity, ambiguity) + F1 (raw & normalized retrieval)
  dissociation.py  arm decomposition, cross-fitted tau, policy value       <-- headline
  analysis.py      effect maps, flip table, harm audit, gate regret, report
  phase1.py        orchestration; run_datasets() across datasets + combined summary
configs/phase1.yaml
notebooks/01_dissociation.ipynb
scripts/run_phase1.py
tests/test_offline.py      # python tests/test_offline.py — 12 tests, no GPU/network
docs/RESEARCH_PLAN.md
```

## Workdir layout (Drive)

```
WORKDIR/
  data/                      sampled query tables + sampling meta
  retrievals/                <dataset>__bm25__k10.jsonl
  logs/gen/                  <dataset>__<model>__arm0.jsonl, ...__bm25_top5__arm1.jsonl
  features/                  joined feature/outcome tables
  results/phase1/<dataset>/  report.md, summary.json, arm_decomposition.csv, tau_evaluation.csv,
                             harm_audit.csv, gate_curve_*.csv, features_binned.parquet, versions.json
  results/phase1_combined/   summary.md, cross_dataset.csv, arm_decomposition_all.csv,
                             tau_evaluation_all.csv
  figures/phase1/<dataset>/  arm_plane.png, tau_evaluation.png, effect_map_informative.png,
                             effect_map_contrast.png, marginals.png, gate_regret.png, delta_hist.png
  pyserini_cache/            persisted BM25 index
```

## Definitions

- Primary effect: `delta_acc = Y1_acc − Y0_acc` (containment). `helped` = Δ>0, `harmed` = Δ<0.
- Continuous outcome: token-F1, max over gold aliases. Reported, but not used to define harm.
- Decision-live: the arms disagree — the only queries a gate can affect. Reported as a diagnostic, never used as a prediction target (see above).
- `D = |AUC(Y1) - 0.5| - |AUC(Y0) - 0.5|`: how much more a feature knows about the retrieval arm than the parametric one.
- Gate regret = oracle value − gate value = harm incurred + benefit forgone (exact identity, tested).

## Troubleshooting

- **`ImportError: bitsandbytes ... requires bitsandbytes>=0.46.1`** → `pip install -U "bitsandbytes>=0.46.1"`, then Runtime → Restart, continue from cell 3.
- **`ModuleNotFoundError: No module named 'pyserini'`** → the quiet install failed; re-run `pip install pyserini` without `-q` and read the error.
- **`NameError: name 'phase1' is not defined`** → you restarted the runtime; re-run cell 3 first.
- **`ImportError: cannot import name '_Ink' from 'PIL._typing'`** → `pip install -q --force-reinstall --no-deps "pillow>=11.3,<12"`, then Runtime → Restart, continue from cell 3.
- **`java not found` / version < 21** → re-run the deps cell; Pyserini needs Java 21.
- **Prebuilt index name error** → the message lists available Wikipedia indexes; set `index_name` in the config.
- **CUDA OOM** → lower `batch_size` (8 → 4).
- **torch downgraded by pip** → Restart runtime, continue from cell 3 without re-running the pip cell.
