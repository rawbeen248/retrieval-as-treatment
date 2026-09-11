"""Phase 1 orchestration. Each step is idempotent and resumable.

    df, meta      = step_data(cfg)
    retr          = step_retrieve(cfg, df)
    gen           = make_generator(cfg)
    gen0          = step_generate(cfg, df, arm=0, gen=gen)
    gen1          = step_generate(cfg, df, arm=1, gen=gen, retr_by_qid=retr)
    feat          = step_features(cfg, df, gen0, gen1, retr)
    out           = step_analysis(cfg, feat, meta, gen.versions)

`run_all(cfg)` chains them. Re-running any step after a disconnect only does
the missing work.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

from . import analysis, score
from .config import Config, paths
from .features import build_feature_table
from .generate import Generator, build_messages, truncate_words
from .logs import JsonlStore


# ------------------------------------------------------------------ paths
def data_path(cfg: Config) -> str:
    return os.path.join(paths(cfg)["data"], f"{cfg.dataset}__n{cfg.n_queries}__seed{cfg.seed}.parquet")


def data_meta_path(cfg: Config) -> str:
    return data_path(cfg).replace(".parquet", ".meta.json")


def retr_store(cfg: Config) -> JsonlStore:
    return JsonlStore(os.path.join(paths(cfg)["retrievals"], f"{cfg.dataset}__{cfg.retriever}__k{cfg.n_retrieve}.jsonl"))


def gen_store(cfg: Config, arm: int) -> JsonlStore:
    if arm == 0:
        name = f"{cfg.dataset}__{cfg.model_tag}__arm0.jsonl"
    else:
        name = f"{cfg.dataset}__{cfg.model_tag}__{cfg.retr_tag}__arm1.jsonl"
    return JsonlStore(os.path.join(paths(cfg)["logs"], name))


def features_path(cfg: Config) -> str:
    return os.path.join(paths(cfg)["features"], f"{cfg.dataset}__{cfg.model_tag}__{cfg.retr_tag}__n{cfg.n_queries}.parquet")


# ------------------------------------------------------------------ steps
def step_data(cfg: Config) -> Tuple[pd.DataFrame, dict]:
    p, mp = data_path(cfg), data_meta_path(cfg)
    if os.path.exists(p):
        df = pd.read_parquet(p)
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        print(f"[data] loaded cached sample: {len(df)} queries from {p}")
        return df, meta
    from .data import load_unified

    df, meta = load_unified(cfg.dataset, cfg.n_queries, cfg.seed, cfg.pop_bins)
    df.to_parquet(p, index=False)
    with open(mp, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[data] sampled {len(df)} queries -> {p}")
    if "per_bin_counts" in meta:
        print(f"[data] per popularity bin: {meta['per_bin_counts']}")
    return df, meta


def step_retrieve(cfg: Config, df: pd.DataFrame) -> Dict[str, List[dict]]:
    store = retr_store(cfg)
    done = store.done_ids()
    todo = df[~df["qid"].astype(str).isin(done)]
    if len(todo):
        from .retrieve import get_retriever, retrieve_all

        retriever = get_retriever(cfg)
        retrieve_all(retriever, todo, cfg.n_retrieve, store)
    else:
        print(f"[retrieve] all {len(done)} queries cached")
    return {q: r["hits"] for q, r in store.load_index().items()}


def make_generator(cfg: Config) -> Generator:
    return Generator(cfg.model_name, cfg.model_revision, cfg.load_in_4bit, cfg.max_prompt_tokens)


def step_generate(cfg: Config, df: pd.DataFrame, arm: int, gen: Optional[Generator] = None,
                  retr_by_qid: Optional[Dict[str, List[dict]]] = None) -> pd.DataFrame:
    assert arm in (0, 1)
    if arm == 1 and retr_by_qid is None:
        raise ValueError("arm 1 needs retr_by_qid")
    store = gen_store(cfg, arm)
    done = store.done_ids()
    todo = df[~df["qid"].astype(str).isin(done)].reset_index(drop=True)
    print(f"[generate arm {arm}] {len(done)} done, {len(todo)} to go -> {store.path}")
    if len(todo) == 0:
        return store.load_df()
    if gen is None:
        gen = make_generator(cfg)

    t0 = time.time()
    for start in tqdm(range(0, len(todo), cfg.batch_size), desc=f"arm {arm}"):
        batch = todo.iloc[start:start + cfg.batch_size]
        msgs, n_used = [], []
        for row in batch.itertuples(index=False):
            if arm == 1:
                hits = (retr_by_qid.get(str(row.qid)) or [])[:cfg.top_k]
                passages = [{"title": h.get("title", ""), "text": truncate_words(h.get("text", ""), cfg.max_passage_words)} for h in hits]
                m, k = gen.fit_messages(row.question, passages)
            else:
                m, k = build_messages(row.question), 0
            msgs.append(m); n_used.append(k)
        outs = gen.generate(msgs, max_new_tokens=cfg.max_new_tokens, prefix_k=cfg.prefix_k)
        recs = []
        for row, o, k in zip(batch.itertuples(index=False), outs, n_used):
            sc = score.score_all(o["answer"], list(row.answers))
            recs.append({"qid": str(row.qid), "arm": arm, "model": cfg.model_name,
                         "retriever": cfg.retr_tag if arm == 1 else "none", "n_passages_used": k,
                         **o, **sc, "ts": time.time()})
        store.append_many(recs)
    dt = time.time() - t0
    print(f"[generate arm {arm}] {len(todo)} generations in {dt/60:.1f} min ({len(todo)/max(dt,1e-9):.2f}/s)")
    return store.load_df()


def step_features(cfg: Config, df: pd.DataFrame, gen0: pd.DataFrame, gen1: pd.DataFrame,
                  retr_by_qid: Dict[str, List[dict]]) -> pd.DataFrame:
    feat = build_feature_table(df, gen0, gen1, retr_by_qid, cfg.top_k)
    p = features_path(cfg)
    feat.to_parquet(p, index=False)
    print(f"[features] {len(feat)} rows x {feat.shape[1]} cols -> {p}")
    return feat


def step_analysis(cfg: Config, feat: pd.DataFrame, data_meta: dict, versions: dict) -> dict:
    P = paths(cfg)
    out = analysis.run_analysis(feat, cfg.to_dict(), versions, data_meta, P["results"], P["figures"],
                                thresholds=cfg.delta_thresholds, harm_thr=cfg.harm_thr,
                                conf_bins=cfg.conf_bins, pop_bins=cfg.pop_bins)
    with open(os.path.join(P["results"], "versions.json"), "w") as f:
        json.dump(versions, f, indent=2)
    print(f"[analysis] report -> {os.path.join(P['results'], 'report.md')}")
    print(f"[analysis] figures -> {P['figures']}")
    return out


def run_all(cfg: Config, skip_generation: bool = False) -> dict:
    df, meta = step_data(cfg)
    retr = step_retrieve(cfg, df)
    versions: dict = {}
    if skip_generation:
        gen0, gen1 = gen_store(cfg, 0).load_df(), gen_store(cfg, 1).load_df()
        vp = os.path.join(paths(cfg)["results"], "versions.json")
        versions = json.load(open(vp)) if os.path.exists(vp) else {}
    else:
        gen = make_generator(cfg)
        gen0 = step_generate(cfg, df, 0, gen=gen)
        gen1 = step_generate(cfg, df, 1, gen=gen, retr_by_qid=retr)
        versions = gen.versions
    feat = step_features(cfg, df, gen0, gen1, retr)
    return step_analysis(cfg, feat, meta, versions)
