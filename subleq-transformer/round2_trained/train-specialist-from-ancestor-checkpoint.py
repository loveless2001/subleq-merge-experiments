#!/usr/bin/env python3
"""
Fine-tune a specialist SUBLEQ transformer from an ancestor checkpoint.

Loads a pre-trained ancestor model and continues training with a
data profile (`ancestor`, `specialist_a`, or `specialist_b`).

Usage:
    python train-specialist-from-ancestor-checkpoint.py \
        --ancestor-checkpoint checkpoints/ancestor/best_model.pt \
        --profile specialist_a \
        --save-dir checkpoints/specialist_a \
        --total-steps 25000

Profiles:
    ancestor      — mixed-data recovery on the broad phase-1 distribution
    specialist_a  — negate, addition, countdown traces (add/sub skills)
    specialist_b  — multiply traces + harder random programs (mul/complex)
"""

import os
import sys
import time
import math
import random
import argparse

import torch
import torch.nn.functional as F

# Add the round2_trained directory to path for subleq imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subleq import MiniSUBLEQTransformer
from subleq.tokenizer import SEQ_LEN, VOCAB_SIZE
from subleq.data import CHANGE_WEIGHT

# Import specialist data generator using importlib since filename has hyphens
import importlib.util
_spec_path = os.path.join(os.path.dirname(__file__), 'subleq', 'specialist-data-generator.py')
_spec = importlib.util.spec_from_file_location('specialist_data_generator', _spec_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pregenerate_specialist_data = _mod.pregenerate_specialist_data


def weighted_cross_entropy(logits, targets, weight_mask):
    """Weighted cross-entropy loss emphasizing changed positions."""
    B, S, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * S, V), targets.reshape(B * S),
                           reduction='none')
    loss = loss.reshape(B, S)
    loss = (loss * weight_mask).sum() / weight_mask.sum()
    return loss


def compute_accuracy(logits, targets, weight_mask=None):
    """Compute full-step and changed-position accuracy."""
    preds = logits.argmax(dim=-1)
    all_correct = (preds == targets).all(dim=1).float().mean().item()

    if weight_mask is not None:
        changed = weight_mask > 1.0
        if changed.any():
            changed_correct = ((preds == targets) | ~changed).all(dim=1).float().mean().item()
        else:
            changed_correct = all_correct
    else:
        changed_correct = all_correct

    return all_correct, changed_correct


def auto_device():
    """Pick best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_specialist(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Profile: {args.profile}")
    print(f"Ancestor: {args.ancestor_checkpoint}")

    # Load ancestor checkpoint
    ckpt = torch.load(args.ancestor_checkpoint, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    ancestor_step = ckpt.get('step', '?')
    ancestor_acc = ckpt.get('best_acc', '?')
    print(f"Ancestor checkpoint: step={ancestor_step}, best_acc={ancestor_acc}")

    # Create model with same architecture as ancestor
    model = MiniSUBLEQTransformer(
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Parameters: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # Cosine decay schedule with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Generate specialist training data
    instr_range = (1, 8)  # Full range since ancestor already covers curriculum
    print(f"Generating {args.data_size} training examples (profile={args.profile})...")
    data_inp, data_out, data_mask = pregenerate_specialist_data(
        args.data_size, profile=args.profile, instr_range=instr_range)
    data_inp = data_inp.to(device)
    data_out = data_out.to(device)
    data_mask = data_mask.to(device)
    print(f"Data ready: {data_inp.shape}")

    best_acc = 0.0
    log_loss = 0.0
    log_full_acc = 0.0
    log_changed_acc = 0.0
    log_count = 0
    start_time = time.time()

    for step_num in range(1, args.total_steps + 1):
        model.train()

        idx = torch.randint(0, data_inp.size(0), (args.batch_size,))
        inp = data_inp[idx]
        out = data_out[idx]
        mask = data_mask[idx]

        logits = model(inp)
        loss = weighted_cross_entropy(logits, out, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            full_acc, changed_acc = compute_accuracy(logits, out, mask)
        log_loss += loss.item()
        log_full_acc += full_acc
        log_changed_acc += changed_acc
        log_count += 1

        if step_num % args.log_every == 0:
            avg_loss = log_loss / log_count
            avg_full = log_full_acc / log_count
            avg_changed = log_changed_acc / log_count
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            print(f"Step {step_num:6d}/{args.total_steps} | "
                  f"loss={avg_loss:.4f} | "
                  f"full_acc={avg_full:.3f} | "
                  f"changed_acc={avg_changed:.3f} | "
                  f"lr={lr:.2e} | "
                  f"time={elapsed:.0f}s")
            log_loss = 0.0
            log_full_acc = 0.0
            log_changed_acc = 0.0
            log_count = 0

        # Regenerate data periodically to avoid overfitting
        if step_num % args.regen_every == 0:
            print(f"  Regenerating data (profile={args.profile})...")
            data_inp, data_out, data_mask = pregenerate_specialist_data(
                args.data_size, profile=args.profile, instr_range=instr_range)
            data_inp = data_inp.to(device)
            data_out = data_out.to(device)
            data_mask = data_mask.to(device)

        # Periodic evaluation
        if step_num % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_inp, eval_out, eval_mask = pregenerate_specialist_data(
                    2000, profile=args.profile, instr_range=(1, 8))
                eval_inp = eval_inp.to(device)
                eval_out = eval_out.to(device)
                eval_mask = eval_mask.to(device)
                eval_logits = model(eval_inp)
                eval_full, eval_changed = compute_accuracy(eval_logits, eval_out, eval_mask)
                eval_loss = weighted_cross_entropy(eval_logits, eval_out, eval_mask).item()

            print(f"  EVAL step {step_num}: loss={eval_loss:.4f} | "
                  f"full_acc={eval_full:.3f} | changed_acc={eval_changed:.3f}")

            if eval_full > best_acc:
                best_acc = eval_full
                ckpt_path = os.path.join(args.save_dir, "best_model.pt")
                torch.save({
                    'step': step_num,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'profile': args.profile,
                    'ancestor_checkpoint': args.ancestor_checkpoint,
                    'config': config,
                }, ckpt_path)
                print(f"  New best: {best_acc:.3f} (saved)")

    # Save final checkpoint
    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({
        'step': args.total_steps,
        'model_state': model.state_dict(),
        'best_acc': best_acc,
        'profile': args.profile,
        'ancestor_checkpoint': args.ancestor_checkpoint,
        'config': config,
    }, final_path)
    print(f"\nTraining complete. Best accuracy: {best_acc:.3f}")
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a specialist SUBLEQ transformer")
    parser.add_argument("--ancestor-checkpoint", type=str, required=True,
                        help="Path to ancestor model checkpoint")
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        choices=['ancestor', 'specialist_a', 'specialist_b'],
        help="Training profile (ancestor, specialist_a, or specialist_b)",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--total-steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-size", type=int, default=100000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--regen-every", type=int, default=2000)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "auto":
        args.device = auto_device()

    os.makedirs(args.save_dir, exist_ok=True)
    train_specialist(args)
