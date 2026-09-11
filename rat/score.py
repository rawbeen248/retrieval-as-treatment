"""Outcome scoring.

Three outcomes are logged for every generation so the analysis can switch
between them without re-running anything:

  f1   token-level F1 (max over gold aliases), SQuAD normalization  -> PRIMARY Y in [0,1]
  em   exact match after normalization (max over aliases)            -> secondary, binary
  acc  PopQA-style containment: any normalized alias is a substring  -> binary, matches
       of the normalized prediction                                     Mallen et al.'s metric
"""
from __future__ import annotations

import collections
import re
import string
from typing import Dict, Iterable, List

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _tokens(s: str) -> List[str]:
    return normalize_answer(s).split()


def f1_single(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_max(pred: str, golds: Iterable[str]) -> float:
    return max((f1_single(pred, g) for g in golds), default=0.0)


def em_max(pred: str, golds: Iterable[str]) -> float:
    npred = normalize_answer(pred)
    return float(any(npred == normalize_answer(g) for g in golds))


def contains_any(pred: str, golds: Iterable[str]) -> float:
    npred = normalize_answer(pred)
    if not npred:
        return 0.0
    for g in golds:
        ng = normalize_answer(g)
        if ng and ng in npred:
            return 1.0
    return 0.0


def score_all(pred: str, golds: Iterable[str]) -> Dict[str, float]:
    golds = list(golds)
    return {
        "f1": f1_max(pred, golds),
        "em": em_max(pred, golds),
        "acc": contains_any(pred, golds),
    }
