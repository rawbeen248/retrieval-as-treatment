"""Generation for both arms, with the F0 confidence probe extracted for free.

Arm 0 (parametric):  system + "Question: ... Answer:"
Arm 1 (treatment):   system + "Use the following passages ... [1] title\\ntext ... Question: ... Answer:"

The answer instruction is identical in both arms; only the passages differ.
That is the treatment.

During arm-0 generation we record, per query, the max-probability and entropy
of the first `prefix_k` generated tokens (the TARG-style prefix probe), the
mean log-prob of the whole answer, its token length and an "I don't know"
flag. These are pre-decision features (tier F0): they come from the parametric
pass that any adaptive system runs anyway.

torch / transformers are imported lazily so the module is importable (and the
pure functions testable) without a GPU.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Reply with the answer only: a short phrase or name. No explanation, no full sentence."
)

PROBE_FIELDS = [
    "probe_maxprob_mean",
    "probe_entropy_mean",
    "first_tok_maxprob",
    "first_tok_entropy",
    "ans_logprob_mean",
    "ans_logprob_sum",
]

_IDK = re.compile(
    r"\b(i do not know|i don't know|don't know|unknown|not sure|cannot (?:be )?determine[d]?|"
    r"no information|unanswerable|not (?:specified|mentioned|provided|available|stated))\b",
    re.I,
)


def truncate_words(text: str, max_words: int) -> str:
    w = (text or "").split()
    return text if len(w) <= max_words else " ".join(w[:max_words])


def format_passages(passages: List[Dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {str(p.get('title', '')).strip()}\n{str(p.get('text', '')).strip()}" for i, p in enumerate(passages)
    )


def build_messages(question: str, passages: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    q = (question or "").strip()
    if passages:
        user = (
            "Use the following passages to help answer the question.\n\n"
            + format_passages(passages)
            + f"\n\nQuestion: {q}\nAnswer:"
        )
    else:
        user = f"Question: {q}\nAnswer:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def clean_answer(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    t = lines[0] if lines else ""
    t = re.sub(r"^\s*answer\s*:\s*", "", t, flags=re.I)
    t = re.sub(r"[.\s]+$", "", t)                       # trailing period(s) before quotes
    t = t.strip().strip('"').strip("'").strip("*").strip()
    t = re.sub(r"[.\s]+$", "", t)                       # and again inside the quotes
    return t


def idk_flag(text: str) -> int:
    return int(bool(_IDK.search(text or "")))


class Generator:
    """Greedy, batched generation with per-step probability statistics."""

    def __init__(self, model_name: str, revision: Optional[str] = None, load_in_4bit: bool = True,
                 max_prompt_tokens: int = 2048):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.model_name = model_name
        self.max_prompt_tokens = max_prompt_tokens

        self.tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kw = dict(revision=revision, device_map="auto")
        prequantized = "bnb-4bit" in model_name.lower() or "bnb_4bit" in model_name.lower()
        if load_in_4bit and not prequantized:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif not prequantized:
            kw["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        self.model.eval()
        self.stop_ids = self._collect_stop_ids()
        self.versions = {
            "model": model_name,
            "revision": revision,
            "commit_hash": getattr(self.model.config, "_commit_hash", None),
            "load_in_4bit": bool(load_in_4bit or prequantized),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "device": str(self.model.device),
        }

    # ------------------------------------------------------------- helpers
    def _collect_stop_ids(self) -> set:
        ids = set()
        gc = getattr(self.model, "generation_config", None)
        for v in (self.tok.eos_token_id, self.tok.pad_token_id,
                  getattr(gc, "eos_token_id", None), getattr(gc, "pad_token_id", None)):
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                ids.update(int(x) for x in v)
            else:
                ids.add(int(v))
        return ids

    def _render(self, messages: List[Dict[str, str]]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def n_tokens(self, messages: List[Dict[str, str]]) -> int:
        return len(self.tok(self._render(messages), add_special_tokens=False)["input_ids"])

    def fit_messages(self, question: str, passages: Optional[List[Dict[str, str]]]) -> Tuple[List[Dict[str, str]], int]:
        """Drop passages from the end until the prompt fits max_prompt_tokens."""
        ps = list(passages or [])
        while True:
            m = build_messages(question, ps if ps else None)
            if not ps or self.n_tokens(m) <= self.max_prompt_tokens:
                return m, len(ps)
            ps = ps[:-1]

    # ---------------------------------------------------------- generation
    def generate(self, messages_list: List[List[Dict[str, str]]], max_new_tokens: int = 32, prefix_k: int = 3) -> List[dict]:
        torch = self.torch
        texts = [self._render(m) for m in messages_list]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tok.pad_token_id,
            )
        in_len = enc["input_ids"].shape[1]
        seqs = out.sequences[:, in_len:]                       # [B, T]
        scores = torch.stack(out.scores, dim=1).float()        # [B, T, V]
        logp = torch.log_softmax(scores, dim=-1)
        p = logp.exp()
        maxp = p.max(dim=-1).values                            # [B, T]
        ent = -(p * logp).sum(dim=-1)                          # [B, T]
        chosen = torch.gather(logp, -1, seqs.unsqueeze(-1)).squeeze(-1)  # [B, T]

        results = []
        for b in range(seqs.shape[0]):
            toks = seqs[b].tolist()
            n = 0
            for t in toks:
                if t in self.stop_ids:
                    break
                n += 1
            raw = self.tok.decode(toks[:n], skip_special_tokens=True)
            rec = {
                "raw": raw,
                "answer": clean_answer(raw),
                "ans_ntokens": int(n),
                "prompt_tokens": int(enc["attention_mask"][b].sum().item()),
                "idk_flag": idk_flag(raw),
            }
            if n > 0:
                k = min(prefix_k, n)
                rec.update(
                    probe_maxprob_mean=float(maxp[b, :k].mean().item()),
                    probe_entropy_mean=float(ent[b, :k].mean().item()),
                    first_tok_maxprob=float(maxp[b, 0].item()),
                    first_tok_entropy=float(ent[b, 0].item()),
                    ans_logprob_mean=float(chosen[b, :n].mean().item()),
                    ans_logprob_sum=float(chosen[b, :n].sum().item()),
                )
            else:
                rec.update({f: float("nan") for f in PROBE_FIELDS})
            results.append(rec)

        del scores, logp, p, maxp, ent, chosen, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results
