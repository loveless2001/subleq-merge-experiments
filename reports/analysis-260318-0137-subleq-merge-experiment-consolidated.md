# SUBLEQ Transformer Merge Experiment — Consolidated Report

**Date:** 2026-03-18
**Workspace:** `/home/lenovo/projects/subleq-merge-experiments`
**Base:** anadim/subleq-transformer Round 2 (6L, 256d, 8H, 1024ff, ~5M params)
**Agents:** Claude (experiment design, merge/alignment code, analysis), Codex (scaffold, training, pipeline execution)

---

## Abstract

We investigated whether two independently trained SUBLEQ transformers — each specialized on different program families — can be merged into a single model that retains both skill sets. Phase 1 (shared ancestor) showed merging works trivially. Phase 2 (independent inits) showed naive merging fails catastrophically, but permutation alignment (Git Re-Basin) followed by a small recovery fine-tune (6K steps, 7.5% of full training cost) restores a strong general executor that exceeds both specialists, given a small additional recovery budget once the specialists already exist. SVD analysis confirms merged models use ~8-9% more of their parameter space than individual specialists.

---

## Phase 1: Shared-Ancestor Merge

**Setup:** Train ancestor (40K steps, broad data) → fork into Specialist A (add/sub traces) and Specialist B (mul/complex traces), each fine-tuned 25K steps → merge.

### Results

| Model | SS% | Neg (201) | Add (300) | Count (20) | Mul (141) | Fib (6) | Rand (100) |
|-------|-----|-----------|-----------|------------|-----------|---------|------------|
| Ancestor 40K | 100 | 201 | 300 | 20 | 141 | **1** | 97 |
| Specialist A | 100 | 201 | 300 | 20 | 141 | 6 | 97 |
| Specialist B | 100 | **197** | 300 | 20 | 141 | **4** | 96 |
| Naive Avg | 100 | 201 | 300 | 20 | 141 | **6** | 97 |
| TIES (k=0.2) | 100 | 201 | 300 | 20 | 141 | **6** | 97 |
| SLERP (t=0.5) | 100 | 201 | 300 | 20 | 141 | **6** | 97 |
| Task Arith (λ=1.0) | 100 | 201 | 300 | 20 | 141 | 5 | 97 |

### Cross-Skill Composition (Option D Follow-up)

| Model | MulAdd (100) | MulNeg (141) |
|-------|-------------|-------------|
| Ancestor 40K | 100% | 96% |
| TIES merge | 100% | **98%** |
| Specialist A | 100% | 95% |

AddNeg was A+A composition (control), not true cross-skill. Ancestor already performed strongly on cross-skill programs, so merged models preserved (not created) composition ability.

### Phase 1 Findings

1. **3/4 merge methods achieved perfect structured scores** — shared-ancestor merge is easy
2. **Merging recovered Fibonacci** (ancestor 1/6 → merged 6/6) — complementary representations combine constructively
3. Specialist B showed expected skill degradation (4 negate failures) — confirms real divergence
4. **Task arithmetic weakest** — full-scale delta addition caused interference at n=6 Fibonacci

### Phase 1 Limitation

Shared ancestor + short fine-tuning kept specialists close in weight space. This is the easiest merge scenario.

---

## Phase 2: Independent-Init Merge

**Setup:** Train Model X (seed 42, specialist_a profile, 80K steps from scratch) and Model Y (seed 123, specialist_b profile, 80K steps from scratch). Same architecture, same 60/40 data structure, different random initializations.

### Results

| Model | SS% | Neg | Add | Count | Mul | Fib | Rand |
|-------|-----|-----|-----|-------|-----|-----|------|
| Indep X (seed42, A) | 99.9 | 201 | 300 | 20 | 141 | 1/6 | 96 |
| Indep Y (seed123, B) | 99.9 | 188 | 300 | 20 | 141 | 0/6 | 97 |
| Naive avg (no align) | **0.0** | 1 | 58 | 0 | 0 | 0 | 0 |
| Aligned avg (Re-Basin) | **0.0** | 1 | 153 | 13 | 0 | 0 | 0 |
| Aligned SLERP | **0.0** | 1 | **300** | **19** | 0 | 0 | 0 |
| **Aligned SLERP + 6K recovery** | **100** | **201** | **300** | **20** | **141** | **6/6** | **97** |

### Phase 2 Findings

1. **Naive merge catastrophically fails** — 0% single-step accuracy. Confirms independent inits break weight-space averaging.
2. **Git Re-Basin (FFN + head permutation) partially recovered some behaviors:**
   - Addition: 58 → 153 (avg) → 300/300 (SLERP) — full recovery
   - Countdown: 0 → 13 (avg) → 19/20 (SLERP)
   - Single-step, multiply, fibonacci remained broken — residual stream misalignment is the bottleneck
3. **SLERP >> linear average after alignment** — geometric interpolation on aligned weights massively outperformed linear mixing
4. **Recovery fine-tuning (6K steps) restored full capability:**
   - 100% single-step, 201/201 negate, 300/300 add, 141/141 multiply, 6/6 fibonacci
   - Best checkpoint at step 6000 (7.5% of 80K full training cost)
   - Recovered model exceeds both individual specialists (6/6 fib vs 1/6 and 0/6)

### Central Contribution

Independent-init models are not directly mergeable. Permutation-style alignment alone is insufficient for full recovery. But alignment followed by a small recovery fine-tune (~6K steps vs 80K from scratch) restores a strong general executor — provided the specialists already exist.

---

## Capacity Analysis: SVD Effective Rank

Effective rank (Shannon entropy of normalized singular values) measures how many weight matrix dimensions actively encode features.

| Model | FFN w1 | FFN w2 | QKV | Attn Out | Token Emb | Out Head |
|-------|--------|--------|-----|----------|-----------|----------|
| Specialist A | 187.9 | 192.4 | 191.8 | 159.8 | 138.6 | 137.4 |
| Specialist B | 188.5 | 193.8 | 192.2 | 159.2 | 138.6 | 139.5 |
| Naive Merge (broken) | **204.9** | **209.3** | **205.8** | **170.9** | 145.6 | 147.8 |
| Recovered (6K) | 200.2 | 204.3 | 202.0 | 167.9 | 143.1 | 146.5 |

### Capacity Findings

1. **Merging increases effective rank by ~8-9%** — specialists ~188-193, merged ~200-209. Merged models utilize more of their parameter space.
2. **Broken and working merged models have similar rank** — the difference is coherence, not capacity. Broken merge has high rank but scrambled features. Recovery organizes them.
3. **Recovery slightly reduced rank** (204.9 → 200.2) — suggests recovery pruned conflicting features, keeping coherent ones. Less rank, more useful rank.

---

## Artifacts

| Item | Path |
|------|------|
| Experiment workspace | `subleq-merge-experiments/` |
| Phase 1 checkpoints | `checkpoints/merged_*` |
| Phase 2 checkpoints | `checkpoints/independent_*` |
| Eval logs | `runs/eval/*.log` |
| Cross-skill eval | `runs/eval/cross_skill_all.log` |
| Phase 1 report | `reports/analysis-260317-1351-subleq-merge-phase1-results.md` |

## Code

| Script | Purpose |
|--------|---------|
| `subleq/specialist-data-generator.py` | Profiled training data (ancestor/A/B) |
| `train-specialist-from-ancestor-checkpoint.py` | Fine-tune from checkpoint |
| `merge-specialist-checkpoints.py` | 4 merge methods (naive, task arith, TIES, SLERP) |
| `align-models-git-rebasin.py` | Weight matching alignment (FFN + heads) |
| `eval-cross-skill-composition.py` | Cross-skill program evaluation |
| `subleq/cross-skill-programs.py` | Composed SUBLEQ programs (mul+add, mul+neg) |

---

## Future Directions

1. **Stronger alignment** — activation matching or iterative d_model permutation to address residual stream bottleneck, potentially eliminating need for recovery fine-tuning
2. **Different architectures** — merge models with different widths/depths (requires architecture-aware alignment)
3. **Scaling** — repeat on larger models/longer programs to test if findings generalize
4. **SVD subspace alignment** — project into shared low-rank feature subspace before merge
