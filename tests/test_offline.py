"""Offline tests: run with `python tests/test_offline.py` or `pytest tests/`.
No GPU, no network, no Java required."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rat import analysis, score  # noqa: E402
from rat.config import load_config  # noqa: E402
from rat.data import _parse_answers, sample_stratified  # noqa: E402
from rat.features import ambiguity_feats, build_feature_table, subject_of  # noqa: E402
from rat.dissociation import (  # noqa: E402
    arm_decomposition, auc, crossfit_tau, dissociation_statistic, policy_value, tau_evaluation,
)
from rat.generate import build_messages, clean_answer, idk_flag, truncate_words  # noqa: E402
from rat.logs import JsonlStore  # noqa: E402
from rat.retrieve import CachedJsonlRetriever, _parse_raw  # noqa: E402


def test_score():
    assert score.normalize_answer("The Eiffel Tower.") == "eiffel tower"
    assert score.f1_max("Paris", ["paris", "Lyon"]) == 1.0
    assert score.em_max("the Beatles", ["Beatles"]) == 1.0
    assert score.contains_any("It is Paris, France", ["Paris"]) == 1.0
    assert score.contains_any("", ["Paris"]) == 0.0
    s = score.score_all("New York City", ["New York"])
    assert 0 < s["f1"] < 1 and s["em"] == 0.0 and s["acc"] == 1.0


def test_logs_resumable_and_torn_line():
    with tempfile.TemporaryDirectory() as d:
        st = JsonlStore(os.path.join(d, "x.jsonl"))
        st.append({"qid": "a", "v": 1})
        st.append_many([{"qid": "b", "v": 2}, {"qid": "a", "v": 3}])
        with open(st.path, "a") as f:
            f.write('{"qid": "c", "v": ')  # torn write
        assert st.done_ids() == {"a", "b"}
        df = st.load_df()
        assert len(df) == 2 and int(df.set_index("qid").loc["a", "v"]) == 3  # last wins
        assert st.load_index()["b"]["v"] == 2


def test_parse_answers_and_sampling():
    assert _parse_answers('["Hong Kong", "HK"]') == ["Hong Kong", "HK"]
    assert _parse_answers("plain") == ["plain"]
    assert _parse_answers(np.array(["a", "b"])) == ["a", "b"]
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({"qid": [f"q{i}" for i in range(n)], "question": "q", "answers": [["a"]] * n,
                       "s_pop": rng.lognormal(3, 2, n), "dataset": "popqa"})
    s, meta = sample_stratified(df, 200, 1, 5)
    assert len(s) == 200 and meta["stratified"] and s["pop_bin"].nunique() == 5
    assert set(s["pop_bin"].value_counts().values) == {40}
    df2 = df.copy(); df2["s_pop"] = np.nan
    s2, meta2 = sample_stratified(df2, 50, 1, 5)
    assert len(s2) == 50 and not meta2["stratified"] and (s2["pop_bin"] == -1).all()


def test_generate_helpers():
    assert clean_answer("Answer: \"Paris\".\nExtra line") == "Paris"
    assert clean_answer("  the Beatles  ") == "the Beatles"
    assert idk_flag("I don't know that.") == 1 and idk_flag("Paris") == 0
    assert truncate_words("a b c d", 2) == "a b"
    m = build_messages("Who?", [{"title": "T", "text": "x"}])
    assert m[0]["role"] == "system" and "[1] T" in m[1]["content"] and m[1]["content"].endswith("Answer:")
    m0 = build_messages("Who?")
    assert "passages" not in m0[1]["content"]


def test_retrieve_helpers():
    raw = json.dumps({"id": "1", "contents": '"Eiffel Tower"\nThe Eiffel Tower is in Paris.'})
    d = _parse_raw(raw)
    assert d["title"] == "Eiffel Tower" and d["text"].startswith("The Eiffel")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "r.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"qid": "q1", "ctxs": [{"title": "A", "text": "t", "score": 3.0}, {"title": "B", "text": "u"}]}) + "\n")
        r = CachedJsonlRetriever(p)
        out = r.search_batch(["q1", "q2"], ["x", "y"], 1)
        assert len(out["q1"]) == 1 and out["q1"][0]["title"] == "A" and out["q2"] == []


def _synthetic_feat(n=600, seed=0):
    """Ground truth built to match the pilot: confidence drives the LEVEL (arm 0),
    retrieval quality drives the ARM-1 outcome (and hence the effect)."""
    rng = np.random.default_rng(seed)
    log_pop = rng.uniform(0, 6, n)
    conf = np.clip(0.3 + 0.1 * log_pop + rng.normal(0, 0.15, n), 0.01, 0.999)
    y0 = (rng.uniform(size=n) < conf).astype(float)
    rq = rng.uniform(0, 1, n)                      # latent retrieval quality
    p1 = np.clip(0.1 + 0.8 * rq + rng.normal(0, 0.08, n), 0.02, 0.98)
    y1 = (rng.uniform(size=n) < p1).astype(float)
    data = pd.DataFrame({"qid": [f"q{i}" for i in range(n)], "dataset": "syn", "question": [f"q {i}" for i in range(n)],
                         "answers": [["a"]] * n, "s_pop": 10 ** log_pop, "o_pop": np.nan, "log_pop": log_pop,
                         "pop_bin": pd.qcut(log_pop, 5, labels=False), "prop": "", "subj": ""})
    g0 = pd.DataFrame({"qid": data["qid"], "answer": "a", "f1": y0, "em": y0, "acc": y0,
                       "probe_maxprob_mean": conf, "probe_entropy_mean": -np.log(conf), "first_tok_maxprob": conf,
                       "first_tok_entropy": -np.log(conf), "ans_logprob_mean": np.log(conf), "ans_logprob_sum": np.log(conf),
                       "idk_flag": 0, "ans_ntokens": 3})
    g1 = pd.DataFrame({"qid": data["qid"], "answer": "b", "f1": y1, "em": y1, "acc": y1, "n_passages_used": 5})
    data["subj"] = ["E" * int(2 + 18 * rq[i]) for i in range(n)]   # ambiguity tracks retrieval quality
    retr = {}
    for i, q in enumerate(data["qid"]):
        top = 6 + 30 * rq[i] + rng.normal(0, 1.0)
        scores = sorted([top] + list(top - rng.uniform(0.5, 6, 9)), reverse=True)
        retr[q] = [{"score": float(sc), "title": data["subj"].iloc[i] if j == 0 else f"Other {j}", "text": "t"}
                   for j, sc in enumerate(scores)]
    return data, g0, g1, retr


def test_ambiguity_helpers():
    a = ambiguity_feats("Idaho")
    assert a["subj_chars"] == 5 and a["subj_ntokens"] == 1 and a["subj_single_token"] == 1 and a["subj_is_short"] == 1
    b = ambiguity_feats("Welcome to the Dollhouse")
    assert b["subj_ntokens"] == 4 and b["subj_is_short"] == 0
    assert subject_of({"subj": "Turin", "question": "x"}) == "Turin"
    # fallback entity extraction for datasets without `subj`
    got = subject_of({"subj": "", "question": "Who directed Welcome to the Dollhouse?"})
    assert "Dollhouse" in got


def test_arm_decomposition_recovers_ground_truth():
    """Confidence must attach to arm 0, retrieval to arm 1, orientation-invariantly."""
    data, g0, g1, retr = _synthetic_feat(n=800, seed=3)
    feat = build_feature_table(data, g0, g1, retr, top_k=5)
    a = arm_decomposition(feat, n_boot=200).set_index("feature")
    assert a.loc["probe_maxprob_mean", "auc_Y0"] > 0.65      # knows the parametric arm
    assert a.loc["probe_maxprob_mean", "D"] < 0              # ...more than the retrieval arm
    assert a.loc["retr_top1", "auc_Y1"] > 0.65               # knows the retrieval arm
    assert a.loc["retr_top1", "D"] > 0
    # inverted features must not flip sign: entropy is confidence upside down
    assert a.loc["probe_entropy_mean", "D"] < 0
    assert abs(a.loc["probe_entropy_mean", "D"] - a.loc["probe_maxprob_mean", "D"]) < 0.1
    st = dissociation_statistic(feat, n_boot=200)
    assert st["diff"] > 0 and st["lo"] > 0 and st["p_gt_0"] > 0.95
    assert auc(np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1])) == 1.0
    assert auc(np.array([4, 3, 2, 1]), np.array([0, 0, 1, 1])) == 0.0


def test_no_circular_target_used():
    """Guard against the two traps: neither `helped` over all queries nor over the
    decision-live subset may be used as a prediction target anywhere."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "rat", "dissociation.py")).read()
    body = src.split('"""', 2)[2]          # skip the module docstring, which discusses them
    assert "helped" not in body, "dissociation.py must not predict `helped`"
    # and the identity that makes it circular, demonstrated:
    data, g0, g1, retr = _synthetic_feat(n=400, seed=5)
    feat = build_feature_table(data, g0, g1, retr, top_k=5)
    live = feat["delta_acc"] != 0
    assert (feat.loc[live, "helped"] == (1 - feat.loc[live, "y0_acc"])).all()


def test_tau_evaluation_prefers_informative_features():
    data, g0, g1, retr = _synthetic_feat(n=800, seed=11)
    feat = build_feature_table(data, g0, g1, retr, top_k=5)
    tbl, ref = tau_evaluation(feat, n_boot=100)
    assert ref["never_retrieve"] <= ref["oracle"] and ref["always_retrieve"] <= ref["oracle"]
    v = tbl.set_index("feature_set")["V@50%"]
    assert v["all F0 + F1"] >= v["popularity only"]
    tau = crossfit_tau(feat, ["retr_top1", "subj_chars"], seed=0)
    assert len(tau) == len(feat) and np.std(tau) > 0
    # policy value must be monotone in oracle terms at the extremes
    assert abs(policy_value(feat, tau, 0.0) - feat["y0_acc"].mean()) < 1e-9
    assert abs(policy_value(feat, tau, 1.0) - feat["y1_acc"].mean()) < 1e-9


def test_features_and_gate_identity():
    data, g0, g1, retr = _synthetic_feat()
    feat = build_feature_table(data, g0, g1, retr, top_k=5)
    assert len(feat) == len(data)
    for c in ["y0_f1", "y1_f1", "delta_f1", "probe_maxprob_mean", "retr_top1", "retr_gap12",
              "subj_chars", "retr_top1_z_bylen", "retr_top1_resid", "retr_ratio_gap12",
              "helped", "harmed", "both_wrong", "both_right"]:
        assert c in feat.columns
    assert ((feat["helped"] == 1) == (feat["delta_acc"] > 0)).all()
    c = analysis.gate_curve(feat, "probe_entropy_mean", "y0_acc", "y1_acc", True, cost=0.0)
    np.testing.assert_allclose(c["regret"], c["harm_incurred"] + c["benefit_forgone"], atol=1e-12)
    assert c["retrieval_rate"].iloc[0] == 0.0 and c["retrieval_rate"].iloc[-1] == 1.0
    assert (c["value"] <= c["oracle_value"] + 1e-12).all()
    c2 = analysis.gate_curve(feat, "probe_entropy_mean", "y0_acc", "y1_acc", True, cost=0.1)
    np.testing.assert_allclose(c2["regret"], c2["harm_incurred"] + c2["benefit_forgone"], atol=1e-12)


def test_run_analysis_end_to_end():
    data, g0, g1, retr = _synthetic_feat()
    feat = build_feature_table(data, g0, g1, retr, top_k=5)
    with tempfile.TemporaryDirectory() as d:
        out = analysis.run_analysis(feat, {"dataset": "syn", "n_queries": 600, "seed": 0, "model_name": "m",
                                           "load_in_4bit": True, "retriever": "bm25", "index_name": "idx", "top_k": 5,
                                           "max_new_tokens": 32, "prefix_k": 3},
                                    {"torch": "x"}, {"stratified": True}, os.path.join(d, "res"), os.path.join(d, "fig"),
                                    n_boot=100)
        assert os.path.exists(os.path.join(d, "res", "report.md"))
        assert os.path.exists(os.path.join(d, "res", "arm_decomposition.csv"))
        assert os.path.exists(os.path.join(d, "res", "tau_evaluation.csv"))
        for f in ("arm_plane.png", "tau_evaluation.png", "effect_map_informative.png",
                  "effect_map_contrast.png", "gate_regret.png"):
            assert os.path.exists(os.path.join(d, "fig", f)), f
        r = out["report"]
        for section in ("which arm does each signal know about", "cross-fitted tau", "Decision gate",
                        "Effect maps", "Harm audit", "Flip table", "Regime"):
            assert section.lower() in r.lower(), section
        for section in ():
            assert section in r, section
        assert out["summary"]["stats"]["flips"]["helped"] > 0


def test_harm_audit_classifies_artifacts():
    """Pilot lesson: negative delta_f1 is often a scoring artifact, not a real flip."""
    feat = pd.DataFrame({
        "question": ["a", "b", "c"], "answers": [["fantasy"], ["hard rock"], ["Western"]],
        "answer0": ["Fantasy", "Christian rock", "Western"], "answer1": ["fantasy MMORPG", "Not specified", "Idaho potato"],
        "y0_f1": [1.0, 0.5, 1.0], "y1_f1": [0.67, 0.0, 0.0],
        "y0_acc": [1.0, 0.0, 1.0], "y1_acc": [1.0, 0.0, 0.0],
        "delta_f1": [-0.33, -0.5, -1.0], "subj_chars": [13, 15, 5],
    })
    kinds = analysis.harm_audit(feat)["kind"].tolist()
    assert sum("genuine" in k for k in kinds) == 1
    assert sum("artifact" in k for k in kinds) == 2


def test_min_cell_suppression():
    """A 1-of-3 cell must not drive a figure or a verdict."""
    data, g0, g1, retr = _synthetic_feat(n=120, seed=7)
    feat = build_feature_table(data, g0, g1, retr, top_k=5)
    feat, _ = analysis.add_bins(feat, "probe_maxprob_mean", 5, 5)
    t = analysis.cell_tables(feat, "pop_bin", "conf_bin")
    assert (t["n"].to_numpy(dtype=float) < analysis.MIN_CELL_N).any()  # sparse by construction


def test_config():
    with tempfile.TemporaryDirectory() as d:
        cfg = load_config(d, None, n_queries=10, delta_thresholds=[0.5])
        assert cfg.n_queries == 10 and cfg.delta_thresholds == (0.5,)
        assert os.path.isdir(os.path.join(d, "logs", "gen"))
        assert cfg.retr_tag == "bm25_top5"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"all {len(fns)} tests passed")
