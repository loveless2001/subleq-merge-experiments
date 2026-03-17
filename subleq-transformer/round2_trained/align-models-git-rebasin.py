#!/usr/bin/env python3
"""
Git Re-Basin weight matching for SUBLEQ transformers.

Aligns model B's neurons to model A's by finding optimal permutations
that minimize weight-space distance. This enables meaningful weight
averaging between independently trained models.

Algorithm: For each layer, solve a linear assignment problem (Hungarian)
to find the permutation of hidden units that best matches B to A.

Permutation groups for this architecture:
1. FFN hidden (d_ff) — independent per layer, permute w1 rows + w2 columns
2. Attention heads — independent per layer, permute head blocks in QKV + out_proj

Note: d_model permutation is skipped due to residual connections constraining
it to be identity (or would require joint optimization across all layers).

Usage:
    python align-models-git-rebasin.py \
        --model-a checkpoints/model_x/best_model.pt \
        --model-b checkpoints/model_y/best_model.pt \
        --output checkpoints/model_y_aligned/aligned_model.pt
"""

import os
import sys
import copy
import argparse

import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subleq import MiniSUBLEQTransformer
from subleq.tokenizer import SEQ_LEN, VOCAB_SIZE


def compute_ffn_cost_matrix(w1_a, b1_a, w2_a, w1_b, b1_b, w2_b):
    """Compute cost matrix for matching FFN hidden units.

    C[i,j] = ||w1_A[i,:] - w1_B[j,:]||^2
           + ||b1_A[i] - b1_B[j]||^2
           + ||w2_A[:,i] - w2_B[:,j]||^2

    w1: (d_ff, d_model), b1: (d_ff,), w2: (d_model, d_ff)
    """
    d_ff = w1_a.shape[0]
    cost = torch.zeros(d_ff, d_ff)

    # w1 row distances: each row i of A vs each row j of B
    # Efficient: ||a_i - b_j||^2 = ||a_i||^2 + ||b_j||^2 - 2 * a_i . b_j
    w1_a_norm = (w1_a ** 2).sum(dim=1)  # (d_ff,)
    w1_b_norm = (w1_b ** 2).sum(dim=1)  # (d_ff,)
    cost += w1_a_norm.unsqueeze(1) + w1_b_norm.unsqueeze(0) - 2 * w1_a @ w1_b.t()

    # b1 distances
    b1_diff = b1_a.unsqueeze(1) - b1_b.unsqueeze(0)  # (d_ff, d_ff)
    cost += b1_diff ** 2

    # w2 column distances
    w2_a_t = w2_a.t()  # (d_ff, d_model)
    w2_b_t = w2_b.t()  # (d_ff, d_model)
    w2_a_norm = (w2_a_t ** 2).sum(dim=1)
    w2_b_norm = (w2_b_t ** 2).sum(dim=1)
    cost += w2_a_norm.unsqueeze(1) + w2_b_norm.unsqueeze(0) - 2 * w2_a_t @ w2_b_t.t()

    return cost


def compute_head_cost_matrix(qkv_a, qkv_b, out_a, out_b, n_heads, d_head):
    """Compute cost matrix for matching attention heads.

    qkv: (3*d_model, d_model), out: (d_model, d_model)
    Each head controls a d_head-sized block in the QKV output and out_proj input.
    """
    cost = torch.zeros(n_heads, n_heads)
    d_model = n_heads * d_head

    for i in range(n_heads):
        for j in range(n_heads):
            # QKV: compare head blocks across all 3 (Q, K, V)
            for qkv_idx in range(3):
                offset = qkv_idx * d_model
                block_a = qkv_a[offset + i * d_head: offset + (i + 1) * d_head, :]
                block_b = qkv_b[offset + j * d_head: offset + (j + 1) * d_head, :]
                cost[i, j] += ((block_a - block_b) ** 2).sum()

            # out_proj: compare column blocks (input side)
            col_a = out_a[:, i * d_head: (i + 1) * d_head]
            col_b = out_b[:, j * d_head: (j + 1) * d_head]
            cost[i, j] += ((col_a - col_b) ** 2).sum()

    return cost


def apply_ffn_permutation(state_dict, layer_idx, perm):
    """Apply permutation to FFN hidden units in model B's state dict.

    Permute w1 rows + b1 entries (output side) and w2 columns (input side).
    Also permute norm2 if it interacts, but in Pre-LN it doesn't
    (norm2 operates on d_model, not d_ff).
    """
    prefix = f'layers.{layer_idx}.ffn'
    state_dict[f'{prefix}.w1.weight'] = state_dict[f'{prefix}.w1.weight'][perm]
    state_dict[f'{prefix}.w1.bias'] = state_dict[f'{prefix}.w1.bias'][perm]
    state_dict[f'{prefix}.w2.weight'] = state_dict[f'{prefix}.w2.weight'][:, perm]
    # w2.bias is (d_model,) — not affected by d_ff permutation


def apply_head_permutation(state_dict, layer_idx, perm, n_heads, d_head):
    """Apply head permutation to attention block in model B's state dict.

    Permute head-sized blocks in QKV weights/biases and out_proj weights.
    """
    prefix = f'layers.{layer_idx}.attn'
    d_model = n_heads * d_head

    # QKV weight: (3*d_model, d_model) — permute head blocks in output dim
    qkv_w = state_dict[f'{prefix}.qkv.weight']
    qkv_b = state_dict[f'{prefix}.qkv.bias']
    new_qkv_w = torch.zeros_like(qkv_w)
    new_qkv_b = torch.zeros_like(qkv_b)

    for qkv_idx in range(3):
        offset = qkv_idx * d_model
        for new_pos, old_pos in enumerate(perm):
            new_qkv_w[offset + new_pos * d_head: offset + (new_pos + 1) * d_head] = \
                qkv_w[offset + old_pos * d_head: offset + (old_pos + 1) * d_head]
            new_qkv_b[offset + new_pos * d_head: offset + (new_pos + 1) * d_head] = \
                qkv_b[offset + old_pos * d_head: offset + (old_pos + 1) * d_head]

    state_dict[f'{prefix}.qkv.weight'] = new_qkv_w
    state_dict[f'{prefix}.qkv.bias'] = new_qkv_b

    # out_proj weight: (d_model, d_model) — permute head blocks in input dim (columns)
    out_w = state_dict[f'{prefix}.out_proj.weight']
    new_out_w = torch.zeros_like(out_w)
    for new_pos, old_pos in enumerate(perm):
        new_out_w[:, new_pos * d_head: (new_pos + 1) * d_head] = \
            out_w[:, old_pos * d_head: (old_pos + 1) * d_head]
    state_dict[f'{prefix}.out_proj.weight'] = new_out_w
    # out_proj.bias is (d_model,) — not affected by head permutation
    # (it's on the output side, which is d_model, not head-indexed)


def align_model_b_to_a(state_a, state_b, n_layers, n_heads, d_head, d_ff):
    """Find and apply optimal permutations to align model B to model A.

    Returns the aligned state dict for model B and alignment statistics.
    """
    aligned_b = copy.deepcopy(state_b)
    stats = {'ffn_costs': [], 'head_costs': []}

    for layer_idx in range(n_layers):
        prefix = f'layers.{layer_idx}'

        # --- FFN alignment ---
        w1_a = state_a[f'{prefix}.ffn.w1.weight'].float()
        b1_a = state_a[f'{prefix}.ffn.w1.bias'].float()
        w2_a = state_a[f'{prefix}.ffn.w2.weight'].float()
        w1_b = aligned_b[f'{prefix}.ffn.w1.weight'].float()
        b1_b = aligned_b[f'{prefix}.ffn.w1.bias'].float()
        w2_b = aligned_b[f'{prefix}.ffn.w2.weight'].float()

        ffn_cost = compute_ffn_cost_matrix(w1_a, b1_a, w2_a, w1_b, b1_b, w2_b)
        row_ind, col_ind = linear_sum_assignment(ffn_cost.numpy())
        ffn_perm = torch.tensor(col_ind, dtype=torch.long)

        # Cost before vs after
        identity_cost = ffn_cost.diagonal().sum().item()
        aligned_cost = ffn_cost[row_ind, col_ind].sum().item()
        stats['ffn_costs'].append((identity_cost, aligned_cost))

        apply_ffn_permutation(aligned_b, layer_idx, ffn_perm)

        # --- Head alignment ---
        qkv_a = state_a[f'{prefix}.attn.qkv.weight'].float()
        qkv_b = aligned_b[f'{prefix}.attn.qkv.weight'].float()
        out_a = state_a[f'{prefix}.attn.out_proj.weight'].float()
        out_b = aligned_b[f'{prefix}.attn.out_proj.weight'].float()

        head_cost = compute_head_cost_matrix(qkv_a, qkv_b, out_a, out_b, n_heads, d_head)
        row_ind, col_ind = linear_sum_assignment(head_cost.numpy())
        head_perm = col_ind.tolist()

        identity_cost = head_cost.diagonal().sum().item()
        aligned_cost = head_cost[row_ind, col_ind].sum().item()
        stats['head_costs'].append((identity_cost, aligned_cost))

        apply_head_permutation(aligned_b, layer_idx, head_perm, n_heads, d_head)

        print(f"  Layer {layer_idx}: FFN perm={ffn_perm.tolist()[:8]}... "
              f"Head perm={head_perm}")

    return aligned_b, stats


def main():
    parser = argparse.ArgumentParser(description="Git Re-Basin alignment for SUBLEQ transformers")
    parser.add_argument("--model-a", type=str, required=True,
                        help="Reference model (alignment target)")
    parser.add_argument("--model-b", type=str, required=True,
                        help="Model to align (will be permuted)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for aligned model B checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("Loading models...")
    ckpt_a = torch.load(args.model_a, map_location=args.device, weights_only=False)
    ckpt_b = torch.load(args.model_b, map_location=args.device, weights_only=False)

    config = ckpt_a.get('config', {})
    n_layers = config.get('n_layers', 6)
    n_heads = config.get('n_heads', 8)
    d_model = config.get('d_model', 256)
    d_head = d_model // n_heads
    d_ff = config.get('d_ff', 1024)

    print(f"Architecture: {n_layers}L, {d_model}d, {n_heads}H, {d_ff}ff")
    print(f"Aligning model B to model A...")

    aligned_state, stats = align_model_b_to_a(
        ckpt_a['model_state'], ckpt_b['model_state'],
        n_layers, n_heads, d_head, d_ff
    )

    # Print alignment stats
    print(f"\nAlignment cost reduction:")
    for i, (before, after) in enumerate(stats['ffn_costs']):
        reduction = (1 - after / before) * 100 if before > 0 else 0
        print(f"  Layer {i} FFN:  {before:.1f} → {after:.1f} ({reduction:.1f}% reduction)")
    for i, (before, after) in enumerate(stats['head_costs']):
        reduction = (1 - after / before) * 100 if before > 0 else 0
        print(f"  Layer {i} Head: {before:.1f} → {after:.1f} ({reduction:.1f}% reduction)")

    # Verify aligned model loads correctly
    model = MiniSUBLEQTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, dropout=0.0,
    )
    model.load_state_dict(aligned_state)
    print(f"\nAligned model loads OK ({model.count_params():,} params)")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'step': ckpt_b.get('step', 0),
        'model_state': aligned_state,
        'best_acc': ckpt_b.get('best_acc', 0),
        'config': config,
        'alignment': {
            'method': 'git_rebasin_weight_matching',
            'reference_model': args.model_a,
            'aligned_model': args.model_b,
            'ffn_cost_reduction': [(b, a) for b, a in stats['ffn_costs']],
            'head_cost_reduction': [(b, a) for b, a in stats['head_costs']],
        },
    }, args.output)
    print(f"Aligned checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
