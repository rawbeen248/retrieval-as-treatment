#!/usr/bin/env python
"""CLI mirror of notebooks/01_dissociation.ipynb.

    python scripts/run_phase1.py --workdir /content/drive/MyDrive/retrieval-as-treatment \
        --datasets popqa triviaqa --n 2000 --n-triviaqa 1500

Re-running only does the missing work (data, retrievals and generations are cached).
"""
import argparse

from rat import phase1
from rat.config import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    p.add_argument("--config", default="configs/phase1.yaml")
    p.add_argument("--datasets", nargs="+", default=["popqa", "triviaqa"],
                   help="popqa, triviaqa, nq_open")
    p.add_argument("--n", type=int, default=None, help="override n_queries for all datasets")
    p.add_argument("--n-popqa", type=int, default=None)
    p.add_argument("--n-triviaqa", type=int, default=None)
    p.add_argument("--n-nq-open", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--skip-generation", action="store_true")
    a = p.parse_args()

    cfg = load_config(a.workdir, a.config, n_queries=a.n, model_name=a.model, batch_size=a.batch_size)
    per = {k: v for k, v in (("popqa", a.n_popqa), ("triviaqa", a.n_triviaqa), ("nq_open", a.n_nq_open)) if v}
    out = phase1.run_datasets(cfg, a.datasets, skip_generation=a.skip_generation, n_by_dataset=per)
    for name, r in out.items():
        print(f"\n\n########## {name} ##########\n")
        print(r["report"])


if __name__ == "__main__":
    main()
