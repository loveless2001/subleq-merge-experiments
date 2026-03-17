"""
Specialist data generation for SUBLEQ merge experiment.

Extends the base data.py to support skill-specific training data:
- Specialist A: negate, addition, countdown traces (add/sub skills)
- Specialist B: multiply traces + harder random programs (mul/complex skills)
- Both share the same random single-step data (core SUBLEQ execution)
"""

import random
import torch

from subleq.interpreter import step, MEM_SIZE
from subleq.tokenizer import encode, SEQ_LEN, get_changed_positions
from subleq.programs import (
    make_negate, make_addition, make_countdown,
    make_multiply, generate_random_state,
    generate_random_program,
)
from subleq.data import generate_step_pair, generate_trace_pairs, generate_batch, CHANGE_WEIGHT


def generate_specialist_trace_batch(count, profile='ancestor',
                                     instr_range=(1, 8),
                                     change_weight=CHANGE_WEIGHT):
    """Generate training pairs from program execution traces, filtered by specialist profile.

    Profiles:
        'ancestor'      — original distribution (broad competence)
        'specialist_a'  — 25% negate, 25% addition, 25% countdown, 25% random
        'specialist_b'  — 30% multiply, 25% random(1-4), 45% random(4-8)
    """
    inputs, outputs, masks = [], [], []

    while len(inputs) < count:
        choice = random.random()

        if profile == 'ancestor':
            # Original distribution from upstream data.py
            if choice < 0.15:
                v = random.randint(-100, 100)
                mem, pc, _ = make_negate(v)
            elif choice < 0.30:
                a, b = random.randint(-60, 60), random.randint(-60, 60)
                mem, pc, _ = make_addition(a, b)
            elif choice < 0.45:
                n = random.randint(1, 20)
                mem, pc, _ = make_countdown(n)
            elif choice < 0.55:
                a = random.randint(1, 10)
                b = random.randint(1, min(12, 127 // max(a, 1)))
                mem, pc, _ = make_multiply(a, b)
            else:
                n_instr = random.randint(*instr_range)
                mem, pc = generate_random_program(n_instr)

        elif profile == 'specialist_a':
            # Add/sub skills: negate, addition, countdown, random programs
            if choice < 0.25:
                v = random.randint(-100, 100)
                mem, pc, _ = make_negate(v)
            elif choice < 0.50:
                a, b = random.randint(-60, 60), random.randint(-60, 60)
                mem, pc, _ = make_addition(a, b)
            elif choice < 0.75:
                n = random.randint(1, 20)
                mem, pc, _ = make_countdown(n)
            else:
                n_instr = random.randint(*instr_range)
                mem, pc = generate_random_program(n_instr)

        elif profile == 'specialist_b':
            # Mul/complex skills: multiply + harder random programs
            if choice < 0.30:
                a = random.randint(1, 10)
                b = random.randint(1, min(12, 127 // max(a, 1)))
                mem, pc, _ = make_multiply(a, b)
            elif choice < 0.55:
                # Random programs with lower instruction count
                n_instr = random.randint(1, 4)
                mem, pc = generate_random_program(n_instr)
            else:
                # Random programs with higher instruction count (harder)
                n_instr = random.randint(4, 8)
                mem, pc = generate_random_program(n_instr)
        else:
            raise ValueError(f"Unknown profile: {profile}")

        pairs = generate_trace_pairs(mem, pc, max_steps=40)
        for inp, out, changed in pairs:
            if len(inputs) >= count:
                break
            inputs.append(inp)
            outputs.append(out)
            mask = torch.ones(SEQ_LEN, dtype=torch.float32)
            for pos in changed:
                mask[pos] = change_weight
            masks.append(mask)

    return (torch.stack(inputs[:count]),
            torch.stack(outputs[:count]),
            torch.stack(masks[:count]))


def pregenerate_specialist_data(size, profile='ancestor', instr_range=(1, 8)):
    """Pre-generate a dataset with specialist profile.

    Mix: 60% random single-step (shared across all profiles), 40% profiled traces.
    The random single-step portion is identical regardless of profile — this ensures
    all specialists learn core SUBLEQ execution, only structured program exposure differs.
    """
    random_count = int(size * 0.6)
    trace_count = size - random_count

    all_inp, all_out, all_mask = [], [], []

    # Random single-step data — shared across all profiles
    bs = 512
    for start in range(0, random_count, bs):
        n = min(bs, random_count - start)
        inp, out, mask = generate_batch(n, instr_range=instr_range)
        all_inp.append(inp)
        all_out.append(out)
        all_mask.append(mask)

    # Profiled trace data — differs by specialist
    if trace_count > 0:
        t_inp, t_out, t_mask = generate_specialist_trace_batch(
            trace_count, profile=profile, instr_range=instr_range)
        all_inp.append(t_inp)
        all_out.append(t_out)
        all_mask.append(t_mask)

    inp = torch.cat(all_inp)
    out = torch.cat(all_out)
    mask = torch.cat(all_mask)

    perm = torch.randperm(inp.size(0))
    return inp[perm], out[perm], mask[perm]
