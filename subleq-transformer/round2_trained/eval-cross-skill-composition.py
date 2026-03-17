#!/usr/bin/env python3
"""
Evaluate cross-skill composition on SUBLEQ transformer checkpoints.

Tests whether merged models can execute programs requiring BOTH
add/sub and mul/complex skills in a single execution trace.

Usage:
    python eval-cross-skill-composition.py checkpoints/merged_naive_average/merged_model.pt
    python eval-cross-skill-composition.py --all-checkpoints  # evaluate all checkpoints
"""

import os
import sys
import argparse
import importlib.util

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subleq import (
    MiniSUBLEQTransformer, run,
    MEM_SIZE, VALUE_MAX,
    encode, decode, SEQ_LEN, VOCAB_SIZE,
)

# Load cross-skill programs via importlib (hyphenated filename)
_spec_path = os.path.join(os.path.dirname(__file__), 'subleq', 'cross-skill-programs.py')
_spec = importlib.util.spec_from_file_location('cross_skill_programs', _spec_path)
_cross = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cross)

make_multiply_then_add = _cross.make_multiply_then_add
make_add_then_negate = _cross.make_add_then_negate
make_multiply_then_negate = _cross.make_multiply_then_negate


def auto_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(path, device='cpu'):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    model = MiniSUBLEQTransformer(
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        dropout=0.0,
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model


def model_step(model, memory, pc, device='cpu'):
    """Use the model to predict one SUBLEQ step."""
    inp = encode(memory, pc).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
    pred_tokens = logits.argmax(dim=-1).squeeze(0)
    new_mem, new_pc = decode(pred_tokens)
    return new_mem, new_pc


def eval_cross_skill(model, device='cpu'):
    """Evaluate cross-skill composition programs."""
    results = {}

    # --- MULTIPLY THEN ADD: a*b + c ---
    print("\n  MULTIPLY_THEN_ADD (a*b + c):")
    correct, total, fails = 0, 0, []
    test_cases = []
    for a in range(1, 8):
        for b in range(1, min(8, VALUE_MAX // max(a, 1) + 1)):
            for c in [0, 1, 5, 10, -5]:
                if a * b + abs(c) <= VALUE_MAX:
                    test_cases.append((a, b, c))

    for a, b, c in test_cases[:100]:
        mem, pc, addr, expected = make_multiply_then_add(a, b, c)
        m, p = list(mem), pc
        for _ in range(200):
            if p < 0 or p + 2 >= len(m):
                break
            m, p = model_step(model, m, p, device)

        total += 1
        if m[addr] == expected:
            correct += 1
        else:
            fails.append((a, b, c, expected, m[addr]))

    print(f"    Result: {correct}/{total} ({100*correct/total:.1f}%)")
    if fails and len(fails) <= 10:
        for a, b, c, exp, got in fails:
            print(f"    FAIL: {a}*{b}+{c} = expected {exp}, got {got}")
    elif fails:
        print(f"    ({len(fails)} failures, showing first 5)")
        for a, b, c, exp, got in fails[:5]:
            print(f"    FAIL: {a}*{b}+{c} = expected {exp}, got {got}")
    results['mul_then_add'] = (correct, total)

    # --- ADD THEN NEGATE: -(a+b) ---
    print("\n  ADD_THEN_NEGATE (-(a+b)):")
    correct, total, fails = 0, 0, []
    test_cases = [(a, b) for a in range(-50, 51, 5) for b in range(-50, 51, 5)
                  if abs(a + b) <= VALUE_MAX]

    for a, b in test_cases[:100]:
        mem, pc, addr, expected = make_add_then_negate(a, b)
        m, p = list(mem), pc
        for _ in range(50):
            if p < 0 or p + 2 >= len(m):
                break
            m, p = model_step(model, m, p, device)

        total += 1
        if m[addr] == expected:
            correct += 1
        else:
            fails.append((a, b, expected, m[addr]))

    print(f"    Result: {correct}/{total} ({100*correct/total:.1f}%)")
    if fails and len(fails) <= 10:
        for a, b, exp, got in fails:
            print(f"    FAIL: -({a}+{b}) = expected {exp}, got {got}")
    results['add_then_negate'] = (correct, total)

    # --- MULTIPLY THEN NEGATE: -(a*b) ---
    print("\n  MULTIPLY_THEN_NEGATE (-(a*b)):")
    correct, total, fails = 0, 0, []
    test_cases = [(a, b) for a in range(1, 13) for b in range(1, min(13, VALUE_MAX // max(a, 1) + 1))
                  if a * b <= VALUE_MAX]

    for a, b in test_cases:
        mem, pc, addr, expected = make_multiply_then_negate(a, b)
        m, p = list(mem), pc
        for _ in range(200):
            if p < 0 or p + 2 >= len(m):
                break
            m, p = model_step(model, m, p, device)

        total += 1
        if m[addr] == expected:
            correct += 1
        else:
            fails.append((a, b, expected, m[addr]))

    print(f"    Result: {correct}/{total} ({100*correct/total:.1f}%)")
    if fails and len(fails) <= 10:
        for a, b, exp, got in fails:
            print(f"    FAIL: -({a}*{b}) = expected {exp}, got {got}")
    elif fails:
        print(f"    ({len(fails)} failures, showing first 5)")
        for a, b, exp, got in fails[:5]:
            print(f"    FAIL: -({a}*{b}) = expected {exp}, got {got}")
    results['mul_then_negate'] = (correct, total)

    return results


def find_all_checkpoints(root):
    """Find all evaluable checkpoints in the experiment workspace."""
    checkpoints = []
    ckpt_dir = os.path.join(root, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return checkpoints
    for name in sorted(os.listdir(ckpt_dir)):
        subdir = os.path.join(ckpt_dir, name)
        for fname in ['best_model.pt', 'merged_model.pt']:
            path = os.path.join(subdir, fname)
            if os.path.exists(path):
                checkpoints.append((name, path))
                break
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Cross-skill composition eval")
    parser.add_argument("model_path", nargs='?', default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Evaluate all checkpoints in workspace")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = auto_device()

    if args.all_checkpoints:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        checkpoints = find_all_checkpoints(root)
        if not checkpoints:
            print("No checkpoints found")
            sys.exit(1)

        all_results = {}
        for name, path in checkpoints:
            print(f"\n{'='*60}")
            print(f"  Evaluating: {name}")
            print(f"{'='*60}")
            model = load_model(path, args.device)
            results = eval_cross_skill(model, args.device)
            all_results[name] = results
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Summary table
        print(f"\n{'='*60}")
        print("  CROSS-SKILL SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Model':<30} {'MulAdd':>10} {'AddNeg':>10} {'MulNeg':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        for name, results in all_results.items():
            ma = results.get('mul_then_add', (0, 0))
            an = results.get('add_then_negate', (0, 0))
            mn = results.get('mul_then_negate', (0, 0))
            ma_pct = f"{100*ma[0]/ma[1]:.0f}%" if ma[1] else "N/A"
            an_pct = f"{100*an[0]/an[1]:.0f}%" if an[1] else "N/A"
            mn_pct = f"{100*mn[0]/mn[1]:.0f}%" if mn[1] else "N/A"
            print(f"  {name:<30} {ma_pct:>10} {an_pct:>10} {mn_pct:>10}")

    elif args.model_path:
        if not os.path.exists(args.model_path):
            print(f"Error: checkpoint not found at {args.model_path}")
            sys.exit(1)
        model = load_model(args.model_path, args.device)
        print(f"Cross-skill evaluation: {args.model_path}")
        eval_cross_skill(model, args.device)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
