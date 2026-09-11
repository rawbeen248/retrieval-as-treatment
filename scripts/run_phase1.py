#!/usr/bin/env python
"""CLI alternative to notebooks/01_harm_map.ipynb.

    python scripts/run_phase1.py --workdir /content/drive/MyDrive/retrieval-as-treatment \
        --config configs/phase1_popqa.yaml --n 2000

Re-running only does the missing work (data / retrieval / generations are all cached).
"""
import argparse

from rat import phase1
from rat.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    p.add_argument("--config", default="configs/phase1_popqa.yaml")
    p.add_argument("--n", type=int, default=None, help="override n_queries")
    p.add_argument("--dataset", default=None)
    p.add_argument("--model", default=None, help="override model_name")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--skip-generation", action="store_true", help="only rebuild features + analysis from cached logs")
    a = p.parse_args()
    cfg = load_config(a.workdir, a.config, n_queries=a.n, dataset=a.dataset, model_name=a.model, batch_size=a.batch_size)
    out = phase1.run_all(cfg, skip_generation=a.skip_generation)
    print(out["report"])


if __name__ == "__main__":
    main()
