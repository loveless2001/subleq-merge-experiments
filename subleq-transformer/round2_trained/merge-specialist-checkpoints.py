#!/usr/bin/env python3
"""
Merge two specialist SUBLEQ transformer checkpoints using various methods.

Supports: naive average, task arithmetic (delta merge), TIES-merging, SLERP.
All methods assume both specialists share the same ancestor checkpoint.

Usage:
    python merge-specialist-checkpoints.py \
        --ancestor checkpoints/ancestor/best_model.pt \
        --specialist-a checkpoints/specialist_a/best_model.pt \
        --specialist-b checkpoints/specialist_b/best_model.pt \
        --method task_arithmetic \
        --output checkpoints/merged_task_arithmetic/merged_model.pt
"""

import os
import sys
import copy
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subleq import MiniSUBLEQTransformer
from subleq.tokenizer import SEQ_LEN, VOCAB_SIZE


def load_state_dict_from_checkpoint(path, device='cpu'):
    """Load model state dict and config from a checkpoint file."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt['model_state'], ckpt.get('config', {})


def merge_naive_average(state_a, state_b):
    """Simple weight average: merged = (A + B) / 2."""
    merged = {}
    for key in state_a:
        merged[key] = (state_a[key].float() + state_b[key].float()) / 2.0
    return merged


def merge_task_arithmetic(state_ancestor, state_a, state_b, scaling=1.0):
    """Task arithmetic: merged = ancestor + scaling * (delta_A + delta_B).

    Delta = specialist - ancestor. Scaling < 1.0 reduces interference.
    """
    merged = {}
    for key in state_ancestor:
        delta_a = state_a[key].float() - state_ancestor[key].float()
        delta_b = state_b[key].float() - state_ancestor[key].float()
        merged[key] = state_ancestor[key].float() + scaling * (delta_a + delta_b)
    return merged


def merge_ties(state_ancestor, state_a, state_b, top_k_pct=0.2, scaling=1.0):
    """TIES-Merging: trim, elect sign, merge.

    1. Compute task vectors (deltas from ancestor)
    2. Trim: keep only top-k% by magnitude, zero the rest
    3. Elect sign: for each parameter, use majority sign across task vectors
    4. Merge: average the sign-aligned, trimmed deltas
    """
    merged = {}
    for key in state_ancestor:
        ancestor_val = state_ancestor[key].float()
        delta_a = state_a[key].float() - ancestor_val
        delta_b = state_b[key].float() - ancestor_val

        # Step 1: Trim — keep only top-k% values by magnitude per task vector
        for delta in [delta_a, delta_b]:
            flat = delta.abs().flatten()
            if flat.numel() == 0:
                continue
            k = max(1, int(flat.numel() * top_k_pct))
            threshold = flat.topk(k).values[-1]
            mask = delta.abs() >= threshold
            delta.mul_(mask.float())

        # Step 2: Elect sign — majority vote
        sign_sum = delta_a.sign() + delta_b.sign()
        elected_sign = sign_sum.sign()
        # Where sign is 0 (conflict), default to delta with larger magnitude
        zero_mask = elected_sign == 0
        if zero_mask.any():
            elected_sign[zero_mask] = torch.where(
                delta_a[zero_mask].abs() >= delta_b[zero_mask].abs(),
                delta_a[zero_mask].sign(),
                delta_b[zero_mask].sign()
            )

        # Step 3: Disjoint merge — keep values that agree with elected sign
        aligned_a = delta_a.clone()
        aligned_b = delta_b.clone()
        aligned_a[delta_a.sign() != elected_sign] = 0
        aligned_b[delta_b.sign() != elected_sign] = 0

        # Average non-zero contributions
        count = (aligned_a != 0).float() + (aligned_b != 0).float()
        count = count.clamp(min=1)
        merged_delta = (aligned_a + aligned_b) / count

        merged[key] = ancestor_val + scaling * merged_delta

    return merged


def merge_slerp(state_a, state_b, t=0.5):
    """Spherical linear interpolation between two model weight vectors.

    t=0.0 gives model A, t=1.0 gives model B, t=0.5 is midpoint on the hypersphere.
    """
    merged = {}
    for key in state_a:
        a = state_a[key].float().flatten()
        b = state_b[key].float().flatten()

        # Normalize
        a_norm = a.norm()
        b_norm = b.norm()

        if a_norm < 1e-8 or b_norm < 1e-8:
            # Degenerate case: fall back to linear interpolation
            merged[key] = ((1 - t) * state_a[key].float() + t * state_b[key].float())
            continue

        a_unit = a / a_norm
        b_unit = b / b_norm

        # Compute angle between vectors
        cos_omega = torch.clamp(torch.dot(a_unit, b_unit), -1.0, 1.0)

        if cos_omega.abs() > 0.9999:
            # Vectors nearly parallel: use linear interpolation
            merged[key] = ((1 - t) * state_a[key].float() + t * state_b[key].float())
            continue

        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega)

        # SLERP on unit vectors, interpolate magnitudes linearly
        interp_unit = (torch.sin((1 - t) * omega) / sin_omega) * a_unit + \
                      (torch.sin(t * omega) / sin_omega) * b_unit
        interp_norm = (1 - t) * a_norm + t * b_norm

        result = interp_unit * interp_norm
        merged[key] = result.reshape(state_a[key].shape)

    return merged


def create_model_from_config(config, device='cpu'):
    """Instantiate a model from config dict."""
    model = MiniSUBLEQTransformer(
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        dropout=0.0,
    )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Merge specialist SUBLEQ checkpoints")
    parser.add_argument("--ancestor", type=str, required=True,
                        help="Path to ancestor checkpoint")
    parser.add_argument("--specialist-a", type=str, required=True,
                        help="Path to specialist A checkpoint")
    parser.add_argument("--specialist-b", type=str, required=True,
                        help="Path to specialist B checkpoint")
    parser.add_argument("--method", type=str, required=True,
                        choices=['naive_average', 'task_arithmetic', 'ties', 'slerp'],
                        help="Merge method")
    parser.add_argument("--scaling", type=float, default=1.0,
                        help="Scaling factor for task arithmetic / TIES (default: 1.0)")
    parser.add_argument("--slerp-t", type=float, default=0.5,
                        help="SLERP interpolation factor (default: 0.5)")
    parser.add_argument("--ties-top-k", type=float, default=0.2,
                        help="TIES top-k percentage to keep (default: 0.2)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for merged checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Loading checkpoints...")
    state_ancestor, config = load_state_dict_from_checkpoint(args.ancestor, args.device)
    state_a, _ = load_state_dict_from_checkpoint(args.specialist_a, args.device)
    state_b, _ = load_state_dict_from_checkpoint(args.specialist_b, args.device)

    print(f"Merging with method: {args.method}")
    if args.method == 'naive_average':
        merged_state = merge_naive_average(state_a, state_b)
    elif args.method == 'task_arithmetic':
        merged_state = merge_task_arithmetic(state_ancestor, state_a, state_b,
                                              scaling=args.scaling)
    elif args.method == 'ties':
        merged_state = merge_ties(state_ancestor, state_a, state_b,
                                   top_k_pct=args.ties_top_k, scaling=args.scaling)
    elif args.method == 'slerp':
        merged_state = merge_slerp(state_a, state_b, t=args.slerp_t)

    # Verify merged state loads into model
    model = create_model_from_config(config, args.device)
    model.load_state_dict({k: v.to(model.output_head.weight.dtype) for k, v in merged_state.items()})
    print(f"Merged model loaded successfully ({model.count_params():,} params)")

    # Save merged checkpoint
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'step': 0,
        'model_state': model.state_dict(),
        'best_acc': 0.0,
        'config': config,
        'merge_method': args.method,
        'merge_params': {
            'scaling': args.scaling,
            'slerp_t': args.slerp_t,
            'ties_top_k': args.ties_top_k,
        },
        'ancestor': args.ancestor,
        'specialist_a': args.specialist_a,
        'specialist_b': args.specialist_b,
    }, args.output)
    print(f"Merged checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
