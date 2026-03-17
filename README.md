# subleq-merge-experiments

Experiment investigating whether independently trained SUBLEQ transformers can be merged into a single model retaining both skill sets. Built on the upstream [subleq-transformer](https://github.com/anadim/subleq-transformer) by anadim.

## Key Results

- **Phase 1 (shared ancestor):** 3/4 merge methods achieve perfect scores. Merging recovered Fibonacci (1/6 → 6/6).
- **Phase 2 (independent init):** Naive merge fails catastrophically (0% accuracy). Git Re-Basin alignment + 6K recovery fine-tuning (7.5% of training cost) restores full capability, exceeding both specialists.
- **SVD analysis:** Merged models use ~8-9% more effective rank than individual specialists.

Full report: `paper/subleq-merge-experiment.tex`

## Layout

```
subleq-merge-experiments/
├── subleq-transformer/          # upstream clone
│   └── round2_trained/          # main experiment code
│       ├── subleq/              # core package (interpreter, model, tokenizer)
│       │   ├── specialist-data-generator.py   # profiled training data
│       │   └── cross-skill-programs.py        # composed SUBLEQ programs
│       ├── train.py                           # upstream training script
│       ├── eval.py                            # upstream evaluation
│       ├── train-specialist-from-ancestor-checkpoint.py  # specialist fine-tuning
│       ├── merge-specialist-checkpoints.py    # 4 merge methods
│       ├── align-models-git-rebasin.py        # Git Re-Basin alignment
│       └── eval-cross-skill-composition.py    # cross-skill evaluation
├── checkpoints/                 # all model checkpoints
│   ├── round2_ancestor_40k/     # Phase 1: ancestor
│   ├── specialist_a/            # Phase 1: add/sub specialist
│   ├── specialist_b/            # Phase 1: mul/complex specialist
│   ├── merged_*/                # Phase 1: merged models
│   ├── independent_*/           # Phase 2: independent-init models + merges
│   └── independent_aligned_slerp_recovery_10k/  # Phase 2: recovered model
├── runs/                        # training + eval logs
│   └── eval/                    # all evaluation logs
├── scripts/                     # wrapper scripts
├── paper/                       # LaTeX paper
│   └── subleq-merge-experiment.tex
└── data/                        # generated datasets
```

## Quick Start

```bash
# Setup
bash scripts/setup_env.sh

# Smoke test
bash scripts/run_round2_smoke.sh

# Phase 1: Shared-ancestor merge
bash scripts/run_round2_ancestor.sh                    # train ancestor (40K steps)
bash scripts/run-specialist-training.sh specialist_a   # train specialist A
bash scripts/run-specialist-training.sh specialist_b   # train specialist B
bash scripts/run-full-merge-experiment.sh              # merge + eval all methods

# Phase 2: Independent-init merge
bash scripts/run-independent-init-training.sh specialist_a 42    # Model X (80K steps)
bash scripts/run-independent-init-training.sh specialist_b 123   # Model Y (80K steps)
bash scripts/run-phase2-align-merge-evaluate.sh                  # align + merge + eval

# Phase 2: Recovery fine-tuning (after alignment)
cd subleq-transformer/round2_trained
python train-specialist-from-ancestor-checkpoint.py \
  --ancestor-checkpoint "$PWD/../../checkpoints/independent_aligned_slerp/merged_model.pt" \
  --profile ancestor \
  --save-dir "$PWD/../../checkpoints/independent_aligned_slerp_recovery_10k" \
  --total-steps 10000

# Evaluation (use absolute paths — eval wrapper cd's into round2_trained)
bash scripts/run_round2_eval.sh "$(pwd)/checkpoints/<name>/best_model.pt"
```

## Merge Methods

| Method | Script Flag | Description |
|--------|-------------|-------------|
| Naive Average | `naive_average` | Simple weight average |
| Task Arithmetic | `task_arithmetic` | Delta merge from shared ancestor |
| TIES-Merging | `ties` | Sign-aware, magnitude-pruned merge |
| SLERP | `slerp` | Spherical linear interpolation |
| Git Re-Basin | `align-models-git-rebasin.py` | Permutation alignment before merge |

## Requirements

- Python 3.10+
- PyTorch >= 2.0.0
- SciPy (for Git Re-Basin alignment)
