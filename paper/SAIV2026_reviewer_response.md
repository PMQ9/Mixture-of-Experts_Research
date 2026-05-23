# SAIV'26 Reviewer Response — Change Log

Mapping of staged revisions in [SAIV2026_submission.tex](SAIV2026_submission.tex) and [references_saiv2026.bib](references_saiv2026.bib) to the reviewer comments in [#23 - SAIV'26.pdf](#23%20-%20SAIV'26.pdf).

---

## Reviewer 23A — Weak accept, Expert

### Q1. Relate metrics (2) and (3) to Casadio et al.'s constraint accuracy / security / satisfaction.

**Change.** Added a "Relation to constraint-based metrics" paragraph immediately after the ExCRA / RoCRA definitions in Sec. *Metrics*.

- `1[Verify(.)]` is recast as a *constraint-satisfaction* verdict.
- ExCRA and RoCRA are recast as *constraint-security* aggregates, over expert correctness and router invariance respectively.
- Clean accuracy is identified with *constraint accuracy*.
- Theorem 1 is then re-read as composing two constraint-security guarantees into a system-level constraint-security guarantee.

**Bib.** Added `casadio2022` (Casadio et al., CAV 2022, pp. 219–231).

### Q2. Clarify practical scenarios for combined CV tasks.

**Change.** Added a lead paragraph *"Why heterogeneous vision MoE?"* at the top of Sec. *Methodology*:

> Deployed vision systems rarely stay within a single training distribution: handling inputs that span disjoint domains is the rule rather than the exception, and dispatching each input to a dataset-specific expert is a natural design. We evaluate on both a visually-dissimilar pair (CIFAR-10 vs. MNIST) and a structurally-similar one (GTSRB+PTSD).

This frames the dataset choice as a deliberate stress test of the framework rather than an arbitrary combination.

### Q3. Declare limitations of epsilon-cube robustness; clarify "ball" vs. "cube" terminology (L∞ makes it a cube).

**Change.** Two places:

1. **Sec. Threat Model.** Kept the literature term "L∞ ball" but added an explicit geometric clarification:

   > We keep the verification-literature term "L∞ ball," but B_ε(x) is geometrically an axis-aligned hypercube of side 2ε; our certified guarantees hold over this cube and not over any wider neighborhood of x.

2. **Sec. Limitations.** Added a new paragraph *"Scope of the certified guarantee"* stating that Theorem 1 certifies each tested cube independently — a *cube-wise* guarantee (Casadio et al.'s constraint security), shared by all current complete verifiers, and silent about unseen images. PGD-AA is positioned as the empirical complement.

3. Also annotated ExCRA / RoCRA as **lower bounds** on true robustness in the metrics section (timeouts and counterexamples both score 0).

### Q4. Whether and how this approach may lead to decomposition of LLMs.

**Change.** Added a fourth direction to Future Work (Sec. *Conclusion and Future Work*):

> Fourth, the same view may scale to MoE LLMs such as Mixtral, DeepSeekMoE, and DeepSeek-V4, which route each token to a sparse subset of hundreds of experts: a port of Theorem 1 would require the soft-routing extension above, distributional rather than argmax specifications, and per-expert verifiers that scale to transformer blocks.

**Bib.** Added `deepseekv4`.

### Minor. p. 3 lists three contributions but they read as parts of one.

**Change.** Removed the standalone "central challenge" paragraph that previously followed Figure 1 and folded its content into the closing sentence of the introduction. The contributions list is reframed as *"three contributions, each building on the previous"* so their logical dependency is explicit rather than reading as three parallel claims.

---

## Reviewer 23B — Weak accept, Knowledgeable

### Weakness. Limited to hard routing (k=1) and disjoint expert class spaces.

**Change.** Sec. *Limitations* restructured. The first paragraph now states the boundary of Theorem 1 upfront in a single sentence:

> Theorem 1 establishes a biconditional only for hard routing (k=1) with disjoint class spaces; soft or top-k routing requires invariance of the full top-k set and routing-weight stability, and overlapping class spaces weaken the result to a one-sided implication.

The router-discrimination caveat (CIFAR/MNIST 100% gating vs. GTSRB/PTSD 93.17%) is folded into the same paragraph so the reader sees both formal and empirical limits in one place.

---

## Reviewer 23C — Accept, Expert

### Eq. 1 normalization.

**Change.** Introduced the *active-expert set*

```
A(c) = { j in {1, ..., k} : c in [o_{s_j}, o_{s_j} + C_{s_j} - 1] }
```

and rewrote Eq. (1) as a per-index convex combination of activated-expert logits:

```
y_hat_c = sum_{j in A(c)}  ( w_j / sum_{l in A(c)} w_l ) * y_{s_j}[c - o_{s_j}]   if A(c) ≠ ∅
        = 0                                                                       otherwise
```

with a note that under hard routing + disjoint classes `|A(c)| <= 1` and Eq. (1) collapses to the previous form `y_hat_c = y_{s_{j*}}[c - o_{s_{j*}}]`. The general form is now the hook for the soft / top-k extension referenced in Sec. *Limitations*.

### Not addressed in this revision

Two of Reviewer 23C's comments would require new experiments and were **not** addressed in this commit:

- **Comparison against monolithic verification.** Would need running alpha-beta-CROWN on a single ONNX graph that encodes the routing decision (likely via an `If` node) and reporting wall-clock + success rate vs. the compositional pipeline.
- **Encoding expert selection in an ONNX graph for monolithic verification.** Related to the above — feasible in principle via gating operators, but would need to be implemented and shown to work.

Worth flagging for camera-ready timeline if reviewers expect a response on these.

---

## Other camera-ready prep (not from reviewers)

- **Figure alt-text.** Added alt-text LaTeX comments to all figures (architecture diagram, three-panel Theorem 1 illustration, router-verification stacked bars, ExCRA heatmap, expert clean-vs-PGD bars, AA-vs-ExCRA grouped bars, training-time stacked bars, latency line plot).
- **Figure height tweaks.** Router verification 3.6 cm → 3.2 cm; latency line plot 3.9 cm → 3.5 cm. Recovers vertical space for the new prose.
- **Conclusion tightened.** Promoted the headline numbers (13.15% AA gap between all-RT and all-NRT experts, 3.2× router training speedup, 100% RoCRA at ε ≤ 4/255) into the opening paragraph, and split Future Work into four directions so the new LLM-decomposition item fits cleanly.
