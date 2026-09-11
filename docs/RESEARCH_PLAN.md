# Retrieval as Treatment: Causal Effect Estimation and Off-Policy Evaluation for Adaptive Retrieval-Augmented Generation

**Working title.** *Retrieval as Treatment: Causal Effect Estimation and Off-Policy Evaluation for Adaptive RAG*
**Alternative.** *Should This Query Retrieve? Treatment-Effect Estimation and Zero-Cost Gate Evaluation for RAG*
**Do not use** "UAR" in any name — "Unified Active Retrieval (UAR)" (Cheng et al., 2024) already occupies it in this exact subfield.

**Target venues.** ARR December 2026 cycle → ACL 2027 (main or Findings); fallbacks: UncertaiNLP 2027, TrustNLP, or an IR venue (SIGIR / CIKM short). The paper is a *method + tool* paper with a small theory section, which fits ACL/EMNLP "efficient methods" and "evaluation" tracks.

**Status.** Design document. No experiments run yet. Every number that appears in the eventual paper must be regenerable from the released code (this rule carried over from the previous project and stays).

---

## 1. Problem statement

### 1.1 In one paragraph

Every retrieval-augmented generation (RAG) system decides, per query, whether to inject retrieved documents into the LLM's context. Retrieval costs tokens and latency, and — critically — it sometimes *lowers* answer quality: the model abandons a correct parametric answer in favour of irrelevant or misleading retrieved text. The decision-relevant quantity is therefore the **change in outcome caused by retrieving**, i.e., a treatment effect. Yet every published adaptive-retrieval method decides using a *proxy* for something else: the LLM's confidence (FLARE, DRAGIN, SeaKR), a query-complexity label (Adaptive-RAG), entity popularity (Mallen et al.), or a reward-weighted classifier over precomputed arms (SLO-conditioned routing). None estimates the effect itself, none offers guarantees on the resulting policy, and every comparison between gating methods requires re-running the LLM for every method on every query. We reformulate adaptive retrieval as **treatment-effect estimation and policy learning under Rubin's potential-outcomes framework**, and we show that a single randomized logged run supports (i) estimating per-query effects, (ii) learning a policy with finite-sample regret guarantees, and (iii) **off-policy evaluation (OPE)** of *any* gating rule — including the published ones — without regenerating a single answer.

### 1.2 Formal statement

Let $q$ be a query with pre-decision features $X(q) \in \mathcal{X}$. Let $A$ be a finite action set; in the base case $A=\{0,1\}$ with $1 = $ "inject top-$k$ passages from retriever $R$" and $0 = $ "answer parametrically". Each action $a$ has a cost $c_a \ge 0$ (normalized tokens/latency). Let $Y^{(a)}(q) \in [0,1]$ be the answer-quality outcome under action $a$.

- **Effect.** $\tau(x) = \mathbb{E}[Y^{(1)} - Y^{(0)} \mid X(q)=x]$ — the conditional average treatment effect (CATE). (We say CATE, not ITE: the estimand is conditional on observable features. See §4.1 for why the distinction matters here.)
- **Policy.** $\pi: \mathcal{X} \to A$. Its value is $V(\pi) = \mathbb{E}\big[Y^{(\pi(X))} - c_{\pi(X)}\big]$.
- **Optimal rule (binary case).** $\pi^*(x) = \mathbb{1}[\tau(x) > c_1 - c_0]$.
- **Three tasks.**
  1. *Estimation*: learn $\hat\tau$ from data with a pre-decision feature restriction (features may not depend on the injected passages; §4.3).
  2. *Policy learning*: choose $\hat\pi \in \Pi$ maximizing an estimate of $V$, with a regret bound $V(\pi^*_\Pi) - V(\hat\pi) = O(\sqrt{\mathrm{complexity}(\Pi)/n})$.
  3. *Off-policy evaluation*: given logged data $\{(x_i, a_i, y_i, e(a_i\mid x_i))\}$ collected under a known logging policy, estimate $V(\pi)$ for any target $\pi$ without new LLM calls, with confidence intervals.

### 1.3 Why proxies are the wrong target (the argument the paper opens with)

Let $\mu_a(x) = \mathbb{E}[Y^{(a)} \mid x]$. Uncertainty gating thresholds an estimate of $\mu_0(x)$ (how likely the parametric answer is wrong). This coincides with $\pi^*$ only if $\tau(x)$ is a monotone function of $\mu_0(x)$ alone — which requires, in particular, that retrieval never hurts ($\mu_1 \ge \mu_0$ everywhere) and that $\mu_1$ is roughly constant. Empirically neither holds: TARG's own appendix reports a quadrant of queries where the model is uncertain **and** retrieval is harmful; Mallen et al. showed retrieval hurts on popular entities. The regret of any proxy gate decomposes exactly into two integrals over the disagreement region — *harm incurred* (retrieved where $\tau<c$) and *benefit forgone* (did not retrieve where $\tau>c$) — which we can measure directly because we observe both arms on the calibration set. This decomposition is a figure in the paper.

---

## 2. Related work (literature survey)

Organized by *what quantity each line of work models*. The point of the section is the table at the end: nobody models $\tau$, nobody offers OPE.

### 2.1 Adaptive retrieval via proxies

- **Popularity / entity frequency.** Mallen et al. (ACL 2023, PopQA) show parametric memory suffices for popular entities and retrieval can introduce noise there; they gate on popularity. First evidence that retrieval has heterogeneous, sometimes negative, effects.
- **Query complexity classification.** Adaptive-RAG (Jeong et al., NAACL 2024) trains a small classifier to predict {no retrieval, single-step, multi-step}, with labels derived from which pipeline got the answer right. Models $P(\text{complexity}\mid q)$; cannot express "retrieval hurts here" as a target. UAR (Cheng et al., 2024) is a related classifier-based trigger.
- **Uncertainty / self-knowledge gates.** FLARE (Jiang et al., EMNLP 2023), DRAGIN (Su et al., ACL 2024), SeaKR (Yao et al., 2024), SKR (Wang et al., 2023), Self-RAG (Asai et al., ICLR 2024) trigger retrieval on token probabilities, entropy, internal states, or learned reflection tokens. All threshold a proxy for $\mu_0$.
- **Training-free gating.** TARG (Nov 2025, arXiv 2511.09803) gates on a $k$-token prefix margin. Its appendix computes $\Delta(q) \in \{-1,0,1\}$ from Always-RAG vs Never-RAG and maps it against uncertainty, finding a harmful-retrieval quadrant — the effect is *measured* but never *modelled* or *used*.
- **Comprehensive benchmark.** Moskvoretskii et al. (ACL 2025) compare 35 methods (8 adaptive-retrieval pipelines, 27 uncertainty estimators) on 6 datasets and find simple uncertainty estimators match complex pipelines at lower cost. Every comparison required full re-execution; there is no offline evaluation tool.

### 2.2 Action routing and bandit framings

- **SLO-conditioned action routing** (Jan 2026, arXiv 2601.00841) selects a retrieval depth per query under cost/refusal/hallucination objectives. It explicitly notes the connection to contextual bandits, counterfactual risk minimization, and doubly-robust estimators, then *precomputes the reward of every action for every query* (full factorial) and trains policies by reward-weighted / best-action classification, leaving counterfactual estimators to future work. This is the closest prior work and it states our gap.
- **Contextual bandits in adjacent settings.** Risk-sensitive bandits for whether a coding agent should use retrieved memory (arXiv 2604.27283, 2026); LinUCB-style multi-LLM selection with regret bounds (arXiv 2506.17670); RouteLLM (Ong et al., 2024) for model routing. Right mathematical family, different decision, and none provides causal identification or OPE for the retrieve-or-not decision.
- **Cost-aware routing** (CA-RAG, arXiv 2606.02581) picks among strategy bundles by a hand-built utility; heuristic, no estimation.

### 2.3 Causal and uplift methods *inside* RAG (must be differentiated)

- **Uplift-RAG** (Qu et al., Findings of EMNLP 2025) defines document utility as the uplift of a *document* over the LLM's internal knowledge and trains a reranker on it. Document-level, at context-construction time, no OPE. Ours is decision-level (whether/which retrieval action) with offline evaluation guarantees. Closest in spirit; cite prominently and differentiate in one sentence.
- **ISFJ-RAG** (Big Data Cogn. Comput. 10(2):56, Feb 2026) uses a structural causal model and counterfactual joint decoding to suppress hallucination inside the decoding loop. **Causal-Counterfactual RAG** (arXiv 2509.14435) integrates counterfactual reasoning into retrieval/generation. Both intervene on the *generation process*; neither treats the retrieval decision as the treatment or evaluates policies.

### 2.4 Retrieval can hurt (empirical grounding for the harm map)

Shi et al. (ICML 2023) on distraction by irrelevant context; Yoran et al. (ICLR 2024) on robustness to irrelevant passages; Cuconasu et al. (SIGIR 2024, "The Power of Noise"); Du et al. (Findings EMNLP 2025) showing context length alone degrades performance even with perfect retrieval; the knowledge-conflict survey (Xu et al., EMNLP 2024). These establish that $\tau(x) < 0$ on a non-trivial region — the region proxy gates cannot represent.

### 2.5 Statistical machinery we import (not contribute)

Potential outcomes (Rubin 1974); AIPW / doubly-robust estimation (Robins, Rotnitzky & Zhao 1994); DR off-policy evaluation (Dudík, Langford & Li, ICML 2011); counterfactual risk minimization (Swaminathan & Joachims 2015); meta-learners for CATE (Künzel et al., PNAS 2019), R-learner (Nie & Wager 2021), DR-learner (Kennedy 2023); empirical welfare maximization and policy-learning regret bounds (Kitagawa & Tetenov, Econometrica 2018; Athey & Wager, Econometrica 2021). We state these results, we do not re-derive them; our contribution is the formulation, the identification argument for the RAG setting, the estimators' empirical behaviour on LLM outcomes, and the tool.

### 2.6 Gap table (goes in the paper)

| Line of work | Models | Uses effect $\tau$? | Guarantees | Evaluates other gates offline? |
|---|---|---|---|---|
| Popularity gate (Mallen 23) | entity frequency | no | no | no |
| Adaptive-RAG (Jeong 24) | $P(\text{complexity}\mid q)$ | no | no | no |
| FLARE / DRAGIN / SeaKR | proxy for $\mu_0$ | no | no | no |
| TARG (Nov 25) | prefix margin; measures $\Delta$ post hoc | measured, not modelled | no | no |
| SLO routing (Jan 26) | reward per arm, full factorial | implicitly, brute force | no | deferred to future work |
| Uplift-RAG (EMNLP 25) | per-document uplift | document-level | no | no |
| **This work** | $\tau(x)$, $V(\pi)$ | **yes** | **regret bound; OPE CIs** | **yes** |

---

## 3. Contributions (what the paper claims)

1. **Formulation.** Adaptive retrieval as treatment-effect estimation and policy learning; a proposition characterizing exactly when proxy gates are optimal, and a regret decomposition (harm vs forgone benefit) that is directly measurable in our setup.
2. **A harm map.** The first systematic measurement of where retrieval *lowers* answer quality, as a function of pre-retrieval signals (base-model confidence × entity popularity), across datasets, two retriever tiers, and two LLMs.
3. **Effect-based gating.** DR-learner CATE estimation from pre-decision features; policies that dominate uncertainty, complexity, and popularity gates on the accuracy-vs-cost frontier at matched retrieval rate; regret bound stated and empirically checked against the observed oracle.
4. **Zero-regeneration gate evaluation (the tool).** A DR off-policy estimator with bootstrap CIs that scores any gating rule from one logged randomized run; validated against ground-truth re-execution for a set of published and synthetic gates; released with the logged dataset so others can benchmark new gates without a GPU.
5. **Multi-arm extension.** Action sets over retrieval depth × retriever × iterative retrieval, where full-factorial evaluation becomes expensive and OPE's cost advantage is large.

---

## 4. Methodology

### 4.1 Honest framing of the causal setting (address this before a reviewer does)

With greedy decoding, a fixed prompt, a fixed retriever, and a fixed corpus, an LLM is a deterministic function: for any single query we *can* run both arms and observe both $Y^{(0)}$ and $Y^{(1)}$. The "fundamental problem of causal inference" does not bind at the level of one query on a research benchmark. So why the causal machinery?

1. **The estimand is still a CATE.** We must decide *before* retrieving, from features $X(q)$ that do not include the retrieved passages. $\tau(x)$ is the effect conditional on those features, and it must be *learned* and *generalized* to unseen queries. That is a CATE-estimation problem regardless of whether $\Delta_i$ is observable on the training set.
2. **Deployment data is single-arm.** Production logs record one action per query. Learning and evaluating policies from such logs is exactly the logged-bandit / OPE regime, and it is the regime in which the tool has value.
3. **Multi-arm action sets make full factorial expensive.** With $|A|=10$ (depths × retrievers × iterative), evaluating a new policy by regeneration costs $10\times$ the queries; OPE costs nothing.
4. **Outcomes become stochastic** under sampling decoding, judge-based scoring, or retriever/corpus updates. The estimators handle this; a deterministic look-up table does not.
5. **Observing both arms is a gift, not a problem.** It gives us *ground truth* to validate CATE estimators and OPE — something medicine and economics never have. We exploit it explicitly (§4.7).

Design consequence: we collect a **full-factorial calibration set** (all arms on every query; provides ground truth) and derive from it a **simulated logged-bandit set** (one arm per query, sampled with known propensity) on which all estimation and OPE methods are run and then checked against the ground truth. A **large single-arm set** (randomized, one arm per query) extends coverage cheaply.

### 4.2 Data

| Dataset | Why | Notes |
|---|---|---|
| **PopQA** | ships entity popularity ($s_{pop}$) and relation type; maximal effect heterogeneity; the harm-map dataset | Mallen et al. 2023 |
| **Natural Questions (open)** | standard single-hop; popular entities; expected mixed effects | KILT/DPR splits |
| **TriviaQA** | strong parametric knowledge for 7B models → many "retrieval unnecessary" and some "retrieval harmful" cases | unfiltered/open |
| **HotpotQA** | multi-hop; the iterative arm matters here; expected large positive $\tau$ | distractor or fullwiki |
| *(optional)* WebQuestions, SQuAD-open | breadth | only if time permits |

Corpus: English Wikipedia (DPR December-2018 100-word passages, or the KILT 2019 snapshot). Prefer **pre-retrieved passages released by prior work** (e.g., Self-RAG's released retrievals for PopQA/TriviaQA; Moskvoretskii et al.'s release if available) to avoid indexing on Colab. Fallback: Pyserini prebuilt BM25 index `wikipedia-dpr-100w`; for the dense tier use a prebuilt dense index or `bge-base-en-v1.5` over a subsampled corpus with FAISS HNSW.

Sizes (initial): 3,000 queries per dataset for the full-factorial calibration set (binary arms), i.e. ~24k generations across four datasets; a 1,000-query-per-dataset subset for the multi-arm factorial. Increase if compute allows.

### 4.3 Treatments, features, costs

**Treatment definition.** $a=1$ means "inject the top-$k$ passages from retriever $R$ into a fixed prompt template." Write $T_R$ when the retriever matters. The effect $\tau$ is a property of the (LLM, retriever, corpus, prompt) tuple, not of the query alone — say so in the paper, and run at least two retriever tiers (BM25 vs dense) to show how the $\tau$ distribution shifts with retriever quality. (Feedback pitfall 3 — adopted.)

**Multi-arm action set** (Phase 4): $A = \{\text{none}, \text{top-1}, \text{top-5}, \text{top-10}, \text{2-round iterative}\} \times \{\text{BM25}, \text{dense}\}$ minus duplicates of "none" → 9 arms. Logging propensity uniform $1/|A|$.

**Pre-decision features $X(q)$.** Features may not depend on the passages that will be injected. (Feedback pitfall 2 — adopted, with a cost-accounting refinement.) Two tiers, both reported:

- **Tier F0 — no retrieval side-effects at all.** Query embedding; query length; named-entity count and type; entity popularity (Wikipedia page views or corpus frequency, matched by string); base-LLM signals from a single parametric forward pass — max token probability and entropy of the first $k$ answer tokens (TARG-style prefix probe), answer length, "I don't know" indicator.
- **Tier F1 — cheap index lookups allowed.** Everything in F0 plus retriever-side scores: BM25 top-1 score, dense top-1 cosine, top-$k$ score dispersion, overlap between BM25 and dense top-$k$. These require an index search but *not* reading passages into the LLM context, which is where the dominant cost (context tokens, generation latency) lies. Report cost accounting separately for search vs generation so the saving claim is precise.

**Costs $c_a$.** Measured average context tokens and wall-clock latency per arm on the stated hardware, normalized to $[0,1]$. The policy objective is $Y - \lambda\,c_a$; we sweep $\lambda$ to trace the accuracy-vs-cost frontier rather than fixing a single cost.

### 4.4 Outcomes

(Feedback pitfall 1 — adopted with a refinement.) Binary exact match is noisy and inflates $\mathrm{Var}(\hat\tau)$. We use:

- **Primary:** token-level F1 against gold aliases, $Y \in [0,1]$.
- **Secondary:** exact match (for comparability with prior work) and a bounded LLM-judge score on a 500-query subset to check that F1-based conclusions hold under semantic scoring. The judge never sees the treatment assignment.
- **Harm definition** for continuous $Y$: $\Delta_i = Y_i^{(1)} - Y_i^{(0)}$; "harmful" if $\Delta_i < -\delta$ with $\delta = 0.5$ (a clear flip), with sensitivity to $\delta \in \{0.25, 0.5, 0.75\}$ reported.

Decoding: greedy for the main results; a temperature-0.7, 3-sample variant on a subset to demonstrate the estimators under genuinely stochastic outcomes.

### 4.5 Estimation of $\tau$

With randomized assignment $e(x) = P(a=1\mid x) = 0.5$ (or $1/|A|$), the assignment is independent of potential outcomes and propensities are known exactly, so inverse-propensity estimators are unbiased regardless of any outcome model, and doubly-robust estimators reduce variance without introducing bias.

Pseudo-outcome for the DR-learner (binary case, $e=0.5$):
$$\hat\phi_i = \hat\mu_1(x_i) - \hat\mu_0(x_i) + \frac{a_i\,(y_i - \hat\mu_1(x_i))}{0.5} - \frac{(1-a_i)\,(y_i - \hat\mu_0(x_i))}{0.5},$$
then regress $\hat\phi_i$ on $x_i$ (cross-fitted) to obtain $\hat\tau(x)$.

Estimators compared: **DR-learner** (primary), **R-learner**, **X-learner**, **T-learner**, **S-learner**, and two non-causal baselines — an **Adaptive-RAG-style classifier** predicting "which arm is correct" and **uncertainty thresholds** on the F0 probe features. Nuisance models: gradient-boosted trees and a small MLP over the same features; the point is that the *estimand*, not the model class, drives the gains. Implementation: EconML `DRLearner` / CausalML meta-learners, with a from-scratch scikit-learn reference implementation released alongside for transparency.

Diagnostics: calibration of $\hat\tau$ (binned mean observed $\Delta$ vs predicted $\hat\tau$ on the full-factorial set); feature ablation F0 vs F1; transfer of $\hat\tau$ trained on one dataset to another (§4.8).

### 4.6 Policy learning

Policy class $\Pi$: thresholded $\hat\tau$ (the plug-in rule), plus direct empirical-welfare maximization over shallow trees and linear policies on $X(q)$ using DR scores as rewards. We state the Kitagawa–Tetenov / Athey–Wager regret bound for the class used and *check it empirically*: on the full-factorial set the oracle policy $\pi^{\text{or}}(q) = \mathbb{1}[\Delta_i > \lambda(c_1-c_0)]$ is observable, so realized regret $V(\pi^{\text{or}}) - V(\hat\pi)$ is a number we can plot against $n$.

Baselines at matched retrieval rate: always-retrieve, never-retrieve, uncertainty gate (prefix entropy / max-prob), popularity gate, Adaptive-RAG-style classifier, TARG-style margin gate, random gate. Report the full accuracy-vs-cost frontier and area under it.

### 4.7 Off-policy evaluation (the tool)

From the logged set $\mathcal{D} = \{(x_i, a_i, y_i, e_i)\}$, for any deterministic target policy $\pi$:
$$\hat V_{DR}(\pi) = \frac{1}{n}\sum_{i=1}^n \Big[\hat\mu_{\pi(x_i)}(x_i) + \frac{\mathbb{1}[a_i = \pi(x_i)]}{e_i}\,\big(y_i - \hat\mu_{a_i}(x_i)\big) - \lambda\,c_{\pi(x_i)}\Big],$$
with bootstrap confidence intervals; also report IPW and direct-method estimates and self-normalized variants for comparison.

**Scope of what can be evaluated.** OPE scores any policy that maps *logged pre-decision features* to a *logged action*. That covers: uncertainty gates reconstructed from the F0 probe (FLARE/DRAGIN/TARG-style *pre-generation* thresholds), popularity gates, Adaptive-RAG-style classifiers, and our own policies. It does **not** cover methods that change the generation process itself (mid-generation retrieval triggers as in full FLARE/DRAGIN, reflection-token decoding as in Self-RAG); those are *additional arms*, and to evaluate them one must log them as arms. State this scope plainly — it is a correction to an over-broad version of the claim.

**Validation protocol (the key figure).** For each of ~10 gates (published-style reconstructions plus synthetic gates spanning the retrieval-rate range): (i) compute $\hat V_{DR}$ from the *simulated single-arm* log; (ii) compute the true $V$ from the *full-factorial* set (ground truth, since both arms are observed); (iii) plot estimated vs true with CIs, report MAE, rank correlation across gates, and CI coverage. Repeat over 200 random maskings to show the estimator's sampling distribution. Cost comparison: LLM calls needed to compare $G$ gates by regeneration ($G \times n$) vs by OPE ($0$ after the one-time log).

### 4.8 Robustness and generality

- **Two LLMs:** Qwen2.5-7B-Instruct (primary), Llama-3.1-8B-Instruct (secondary); optionally a 1.5–3B model to show effect heterogeneity grows as parametric knowledge shrinks.
- **Two retriever tiers:** BM25 vs dense; report $\tau$ distributions and harm rates per tier.
- **Transfer:** train $\hat\tau$ on dataset A, evaluate policy value on dataset B via both OPE and ground truth; quantify degradation. OPE validity requires target queries drawn from the logging distribution; when they are not, report importance-reweighted estimates and flag as a limitation.
- **Stochastic outcomes:** temperature-sampling subset.
- **Sensitivity:** $\delta$ for harm, $\lambda$ for cost, $k$ for top-$k$, ECE-style calibration of $\hat\tau$.

### 4.9 Theory section (short, honest)

- **Prop. 1** (optimal rule): $\pi^*(x) = \arg\max_a \mu_a(x) - \lambda c_a$; binary case $\mathbb{1}[\tau(x) > \lambda(c_1 - c_0)]$.
- **Prop. 2** (when proxies are optimal): a gate of the form $\mathbb{1}[g(\mu_0(x)) > t]$ equals $\pi^*$ for some $t$ iff $\tau$ is a monotone function of $\mu_0$ on the support; a sufficient violation is any region with $\mu_1 < \mu_0$. Corollary: the regret of a proxy gate equals $\int_{\text{harm}} (\lambda\Delta c - \tau)\,dP + \int_{\text{forgone}} (\tau - \lambda\Delta c)\,dP$.
- **Identification:** under randomized logging with known $e$, $\tau$ and $V(\pi)$ are identified; DR estimators are unbiased and their variance is stated.
- **Regret:** cite and state the EWM bound for the policy class used.
No new theorems are claimed; the novelty is the formulation and the empirical programme.

---

## 5. Things to do, in order

### Phase 0 — Setup (days 1–3)
1. Create repo skeleton (§7). Pin versions. Seed everything.
2. Load PopQA, NQ-open, TriviaQA, HotpotQA; unify to `(qid, question, answers, dataset, meta)`; keep PopQA `s_pop` and relation type.
3. Obtain retrievals: prefer released top-$k$ passages; else Pyserini BM25 on `wikipedia-dpr-100w`; dense tier via prebuilt index or bge-base on a subsample. Cache top-10 per query per retriever to disk.
4. Load Qwen2.5-7B-Instruct in 4-bit; fix prompt templates for arm 0 and arm 1; greedy decoding; `max_new_tokens=32`. Smoke test on 50 queries.
5. Implement scoring: token-F1, EM; wire the judge for the subset later.

### Phase 1 — Day-1 experiment: the harm map (days 3–5)
6. PopQA, 2,000 queries, both binary arms, BM25 tier. Record $Y^{(0)}, Y^{(1)}$, prefix-probe features, popularity.
7. Plot the harm map: $x$ = base-model prefix confidence, $y$ = $\log s_{pop}$, colour = fraction harmful ($\Delta < -0.5$); overlay fraction helpful. Also the 1-D marginals.
8. Compute the proxy-gate regret decomposition (harm incurred vs benefit forgone) for an uncertainty gate swept over thresholds. If the harmful fraction is non-trivial in any cell — expected on popular entities — the motivating figure exists. **Decision gate: proceed only if this holds; if not, switch to a smaller LLM where parametric knowledge is weaker and re-check.**

### Phase 2 — Full-factorial calibration set (week 2)
9. Extend to 3,000 queries × 4 datasets × 2 arms (BM25 tier). Then the dense tier on the same queries.
10. Extract F0 and F1 features for every query; freeze feature tables.
11. Report per-dataset: mean $\Delta$, harm rate, help rate, and how they move from BM25 to dense.

### Phase 3 — Estimation and policy learning (weeks 3–4)
12. Simulate the logged-bandit set by masking one arm per query at random ($e=0.5$); keep 200 masking seeds.
13. Fit DR-, R-, X-, T-, S-learners and the two non-causal baselines on F0 and on F1; cross-fitting; calibration plots of $\hat\tau$ vs observed $\Delta$.
14. Derive policies; compute accuracy-vs-cost frontiers against all baselines at matched retrieval rate; plot realized regret vs $n$ against the observed oracle.
15. Transfer experiments across datasets.

### Phase 4 — OPE tool and validation (weeks 4–5)
16. Implement DR / IPW / DM / SN-IPW estimators with bootstrap CIs as a small library with a one-call API: `evaluate(policy_fn, log) -> (value, ci)`.
17. Reconstruct ~6 published-style gates from F0 features (prefix-entropy, max-prob, margin, popularity, complexity classifier, TARG-style) plus ~4 synthetic gates spanning retrieval rates.
18. Validation figure: estimated vs true value, MAE, rank correlation, CI coverage over masking seeds; LLM-call cost comparison.
19. Stochastic-outcome subset (temperature sampling) and judge-score subset; repeat validation.

### Phase 5 — Multi-arm extension (weeks 5–6)
20. 1,000 queries per dataset × 9 arms (depths × retrievers × 2-round iterative). Full factorial for ground truth; simulated uniform logging.
21. Multi-treatment DR-learner and multi-arm OPE; policy over arms; show OPE cost advantage scales with $|A|$.
22. HotpotQA analysis: where the iterative arm has large positive $\tau$ and where it doesn't.

### Phase 6 — Second LLM, sensitivity, theory (week 7)
23. Repeat Phases 2–4 core runs on Llama-3.1-8B-Instruct (smaller $n$ acceptable).
24. Sensitivity: $\delta$, $\lambda$, $k$.
25. Write the theory section (§4.9) and check Prop. 2's decomposition numerically against Phase 1 results.

### Phase 7 — Paper, release, submission (weeks 8–10)
26. Figures: (F1) harm map; (F2) proxy-gate regret decomposition; (F3) accuracy-vs-cost frontiers; (F4) $\hat\tau$ calibration; (F5) OPE estimated-vs-true with CIs; (F6) realized regret vs $n$; (F7) BM25 vs dense $\tau$ shift; (F8) multi-arm OPE cost scaling. Tables: dataset stats; estimator comparison; gate comparison via OPE and via truth; cost accounting. No figure re-plots a table.
27. Release: code, logged datasets (features + arm + outcome + propensity), the OPE library, a notebook that reproduces every figure from the logs without a GPU.
28. Limitations and ethics; reproducibility statement with hardware and seeds. Fresh-environment reproduction run before submission.

---

## 6. Evaluation metrics (definitions fixed up front)

- **Outcome:** token-F1 (primary), EM, judge score (subset).
- **Effect statistics:** mean $\Delta$; harm rate $P(\Delta < -\delta)$; help rate $P(\Delta > \delta)$; by dataset, retriever, confidence bin, popularity bin.
- **Policy quality:** accuracy (mean $Y$) at retrieval rate $r$; area under the accuracy-vs-retrieval-rate curve; welfare $\mathbb{E}[Y - \lambda c]$; realized regret vs observed oracle.
- **Estimator quality:** $\hat\tau$ calibration (binned), RMSE of $\hat\tau$ vs $\Delta$ on held-out full-factorial queries, feature-ablation deltas.
- **OPE quality:** MAE of $\hat V$ vs true $V$ across gates; Spearman rank correlation of gates; 95% CI empirical coverage; LLM calls saved.
- **Cost:** mean context tokens and latency per arm; search vs generation separated.

---

## 7. Repository layout

```
retrieval-as-treatment/
  README.md
  requirements.txt              # pinned
  pyproject.toml                # `pip install -e .` installs the `rat` package
  configs/                      # datasets, models, retrievers, arms, seeds (YAML)
  docs/                         # this plan
  rat/                          # the package
    config.py                   # Config dataclass, YAML loading, workdir layout
    data.py                     # unified loaders + popularity-stratified sampling
    retrieve.py                 # BM25 (Pyserini prebuilt Wikipedia) / dense tiers
    generate.py                 # arm execution, greedy + sampled, confidence probe
    score.py                    # F1 / EM / containment / judge
    logs.py                     # resumable JSONL store
    features.py                 # F0 / F1 extraction joined with outcomes
    analysis.py                 # harm map, regret decomposition, figures, report   (Phase 1)
    phase1.py                   # Phase 1 orchestration
    cate.py                     # DR/R/X/T/S learners + baselines                  (Phase 3)
    policy.py                   # plug-in and EWM policies                          (Phase 3)
    ope.py                      # DR / IPW / DM / SN-IPW + bootstrap CIs  <-- the tool (Phase 4)
    gates.py                    # published-style gate reconstructions              (Phase 4)
  scripts/run_phase1.py         # CLI mirror of the notebook
  tests/test_offline.py         # no-GPU tests (scoring, logs, sampling, analysis identity)
  notebooks/
    01_harm_map.ipynb
    02_factorial_collection.ipynb
    03_cate_and_policy.ipynb
    04_ope_validation.ipynb
    05_multiarm.ipynb
    06_figures.ipynb            # reproduces every figure from data/logs, CPU only
  paper/
```

### 7.1 Running on Google Colab

The layout above is the GitHub repo. Colab is the execution environment. The split is:

- **GitHub** holds `rat/`, `configs/`, `notebooks/`, `requirements.txt`. Nothing generated is committed.
- **Google Drive** (`MyDrive/retrieval-as-treatment/`) holds everything generated: `retrievals/`, `features/`, `logs/`, `figures/`. This is what survives disconnects and what you eventually publish.
- **Each notebook** starts with the same bootstrap cell: mount Drive, clone or pull the repo, install requirements + `pip install -e .`, set `WORKDIR` to the Drive path. Notebooks contain orchestration and plots only; all logic is in `rat/` so it is testable and reusable across notebooks.

**Bootstrap cell (identical at the top of every notebook):**

```python
from google.colab import drive; drive.mount('/content/drive')
WORKDIR = '/content/drive/MyDrive/retrieval-as-treatment'
import os, subprocess
if not os.path.exists('/content/rat'):
    subprocess.run(['git', 'clone', 'https://github.com/<you>/retrieval-as-treatment', '/content/rat'])
else:
    subprocess.run(['git', '-C', '/content/rat', 'pull'])
!pip install -q -r /content/rat/requirements.txt && pip install -q -e /content/rat
import sys; sys.path.insert(0, '/content/rat')
from rat.config import load_config; cfg = load_config(WORKDIR, path='/content/rat/configs/phase1_popqa.yaml')
```

**Resumability rule (non-negotiable).** Every generation run writes one JSONL line per `(qid, arm, retriever, model)` to Drive *as it completes*, and on start reads the existing file and skips finished keys. A Colab disconnect at 60% then costs nothing. Never hold results only in memory; never write a single big file at the end.

**Which phases need a GPU.**

| Phase | Needs GPU? | Rough cost on T4 / on L4–A100 |
|---|---|---|
| 0 setup, retrieval caches | No (BM25) / small GPU for dense encoding | 1–2 h once |
| 1 harm map (2k × 2 arms) | Yes | ~1 h / ~15 min |
| 2 full factorial (12k × 2 arms × 2 retrievers ≈ 48k gens) | Yes | ~8–12 h / ~1.5–2.5 h |
| 3 CATE + policy | **No** — sklearn / EconML, CPU | minutes |
| 4 OPE tool + validation | **No** — CPU | minutes |
| 5 multi-arm (4k × 9 arms ≈ 36k gens) | Yes | ~6–9 h / ~1–2 h |
| 6 second LLM | Yes | ~half of Phase 2 |
| 7 figures, paper | **No** | — |

Estimates assume Qwen2.5-7B-Instruct in 4-bit, batched generation, prompts of ~600–1,000 tokens with 5 passages, `max_new_tokens=32`. Roughly half the total GPU time is Phase 2 and it is embarrassingly resumable, so it can be spread across several free-tier sessions. **Colab Pro** (L4 or A100, longer sessions) collapses the whole project's GPU time to well under a day and is the recommended spend; the free T4 tier works but expect Phase 2 to take three or four sessions.

**Practical constraints to plan around.**

- *Memory.* 7B in 4-bit needs ~6 GB VRAM; T4 (16 GB) is fine with batch size 8–16 at 1k-token prompts. Llama-3.1-8B is similar. Do not attempt a flat dense index over full Wikipedia on Colab RAM — use released top-$k$ retrievals, a prebuilt index, or dense retrieval over a subsampled corpus with FAISS HNSW.
- *Pyserini needs Java.* `apt-get install -y openjdk-21-jdk` in the setup cell; the prebuilt `wikipedia-dpr-100w` BM25 index is a one-time ~2 GB download — cache it on Drive, not `/content`.
- *Model downloads.* Set `HF_HOME` to a Drive path so 7B weights (~5 GB in 4-bit-loadable form) are not re-downloaded every session. Store the HF token in Colab Secrets, not in the notebook.
- *Session hygiene.* Run Phase 2 as a loop over `(dataset, retriever, arm)` chunks of ~500 queries; each chunk is a checkpoint. Keep a small `status.json` on Drive listing completed chunks so a fresh session can print what remains.
- *CPU phases can run anywhere.* Phases 3, 4 and 7 read only `features/` and `logs/` from Drive; run them in a CPU Colab runtime, or download those folders and run locally — this is also the "reproduce every figure without a GPU" path promised in the release.
- *Determinism.* Greedy decoding plus fixed seeds gives reproducible outputs for a given model revision and library version; pin `transformers`, `bitsandbytes`, and the HF model revision hash in `configs/`, and record them in every log line.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Harm rate turns out negligible for the 7B model | Use a smaller LLM (weaker parametric knowledge makes effects more heterogeneous, not less); use the dense tier where distractors are more plausible; PopQA popular entities are the most likely harm region — check there first |
| Reviewer: "you can just run both arms" | §4.1 argument; multi-arm extension; single-arm deployment framing; stochastic-outcome subset; and the both-arms set is what *validates* the estimators |
| Reviewer: "Adaptive-RAG with a different label" | Estimand (difference vs level), estimator (DR vs classifier), guarantees, OPE — plus the regret decomposition showing what the label misses |
| Reviewer: "Uplift-RAG already did uplift" | Document-level reranking vs decision-level policy + OPE; cite and differentiate in one sentence |
| OPE estimates biased because gates use features not in the log | Log a superset of features up front (F1 tier); state the scope limitation (§4.7) |
| Judge noise / EM noise | Continuous F1 primary; judge subset; report variance |
| Colab compute | Pre-retrieved passages; 4-bit; batch generation; cache everything; multi-arm on 1k subset only |
| Distribution shift between log and target | Transfer experiments; importance reweighting; explicit limitation |
| Dataset/model availability drift | Pin versions; defensive loaders; record exact HF revisions |

---

## 9. Limitations to state in the paper

$\tau$ is a property of a (model, retriever, corpus, prompt) tuple, not of queries in the abstract; harm maps do not transfer across models without re-logging. OPE is valid for the logged action set and logged feature superset only, and for target queries from the logging distribution. Greedy decoding makes outcomes deterministic on benchmarks; the stochastic subset partially addresses this. English-only; short-answer QA; 7–8B open models. No claims about generation quality relative to larger LLMs.

---

## 10. Feedback triage (what was adopted, refined, or rejected)

| Feedback item | Decision | Reason |
|---|---|---|
| Add Uplift-RAG and ISFJ-RAG to related work | **Adopted** (both verified to exist) | Closest causal/uplift work inside RAG; must be differentiated |
| Frame as ITE | **Refined → CATE** | The estimand is conditional on features; and with greedy decoding per-query effects are observable — which we turn into a validation asset (§4.1) rather than hide |
| Randomized $e=0.5$ "eliminates propensity error, Neyman-orthogonal, oracle-rate" | **Adopted, stated more modestly** | Known propensity ⇒ IPW unbiased, DR variance-reduced; convergence of $\hat\tau$ still depends on nuisance quality |
| Continuous $Y$ instead of EM | **Adopted** (token-F1 primary, EM secondary, judge subset) | Reduces variance; EM kept for comparability |
| Pre-retrieval-only features | **Adopted with F0/F1 tiers and explicit cost accounting** | Index lookup is cheap; context injection is the cost; keep the saving claim precise |
| Retriever dependence; two tiers | **Adopted** | $\tau$ is retriever-specific |
| Harm map on confidence × popularity | **Adopted** as Day-1 experiment and Figure 1 | |
| DR vs S/T-learners vs classifier | **Adopted, extended with R- and X-learners** | |
| "Validate OPE on FLARE, DRAGIN, Adaptive-RAG" | **Refined** | Only *pre-generation* reconstructions of these gates are evaluable from logs; mid-generation triggers are arms, not gates (§4.7) |
| "Finite-sample regret bounds bridge NLP and learning theory" | **Adopted as stated-and-checked, not derived** | Bounds are imported (Kitagawa–Tetenov; Athey–Wager); we check them empirically against the observed oracle |
| Claim of "foundational reference" impact | **Not written into the paper** | Let reviewers decide; overclaiming is how the previous project got rejected |
