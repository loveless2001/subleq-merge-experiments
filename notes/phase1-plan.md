# Phase 1 Plan

Locked scope:
- Use `subleq-transformer/round2_trained` as the base.
- Reproduce the upstream baseline before changing the data recipe.
- Train one shared ancestor checkpoint.
- Fine-tune two sibling specialists from that ancestor.
- Compare naive averaging, delta merge, and a conflict-aware merge.

Current repo findings:
- Training entrypoint: `subleq-transformer/round2_trained/train.py`
- Evaluation entrypoint: `subleq-transformer/round2_trained/eval.py`
- Data recipe hook: `subleq-transformer/round2_trained/subleq/data.py`
- Checkpoint path is already configurable through `--save-dir`

Immediate next steps:
1. Run `bash scripts/run_round2_smoke.sh` to verify the local wrapper and output paths.
2. Run `bash scripts/run_round2_ancestor.sh` to produce the shared ancestor checkpoint.
3. Add an experiment-owned data-profile path for `ancestor`, `specialist_a`, and `specialist_b`.
4. Add checkpoint initialization so sibling fine-tunes can start from the ancestor.
5. Add a result collector that turns upstream eval output into a comparison table.
