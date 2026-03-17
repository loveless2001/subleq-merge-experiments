"""
Cross-skill SUBLEQ programs that require both add/sub and mul/complex
capabilities in a single execution trace.

These test whether merged models achieve genuine skill integration
vs merely preserving non-interfering skill islands.

Memory layout: cells 0-23 = code (8 instructions), cells 24-31 = data.
Values: 8-bit signed (-128 to 127).
"""

from subleq.interpreter import MEM_SIZE, VALUE_MAX, clamp


def make_multiply_then_add(a, b, c):
    """Compute a*b + c using multiply loop then addition.

    Phase 1 (multiply): result = a*b via repeated addition (uses Specialist B pattern)
    Phase 2 (add): result += c via subtraction of negated c (uses Specialist A pattern)

    Data layout: 24=-a, 25=counter(b), 26=result, 27=const1, 28=c_neg, 29=temp
    Instructions: 0-8 = multiply loop, 9-17 = add c, 18-20 = halt
    """
    a_val, b_val = abs(clamp(a)), abs(clamp(b))
    c_val = clamp(c)
    assert a_val * b_val + abs(c_val) <= VALUE_MAX, f"Overflow: {a_val}*{b_val}+{c_val}"

    mem = [0] * MEM_SIZE

    # Phase 1: Multiply loop (a*b) — same pattern as make_multiply
    # Instr 0 (pc=0): result -= (-a) → result += a
    mem[0] = 24; mem[1] = 26; mem[2] = 3
    # Instr 1 (pc=3): counter -= 1, if <=0 goto phase 2 (pc=9)
    mem[3] = 27; mem[4] = 25; mem[5] = 9
    # Instr 2 (pc=6): trampoline back to 0
    mem[6] = 29; mem[7] = 29; mem[8] = 0

    # Phase 2: Add c — negate-and-subtract pattern
    # Instr 3 (pc=9): clear temp
    mem[9] = 29; mem[10] = 29; mem[11] = 12
    # Instr 4 (pc=12): temp -= c_neg → temp = c
    mem[12] = 28; mem[13] = 29; mem[14] = 15
    # Instr 5 (pc=15): result -= temp → result -= c... wait, we want result += c
    # So we store -c in cell 28, then: temp = -(-c) = c, result -= c would subtract
    # Actually: store c in cell 28, temp -= c → temp = -c, result -= (-c) = result + c
    mem[15] = 29; mem[16] = 26; mem[17] = 18

    # Instr 6 (pc=18): halt
    mem[18] = 29; mem[19] = 29; mem[20] = -1

    # Data
    mem[24] = clamp(-a_val)   # -a for multiply loop
    mem[25] = b_val            # counter
    mem[26] = 0                # result (accumulator)
    mem[27] = 1                # constant 1
    mem[28] = c_val            # c value (for add phase)
    mem[29] = 0                # temp

    expected = a_val * b_val + c_val
    return mem, 0, 26, expected


def make_add_then_negate(a, b):
    """Compute -(a+b): first add, then negate the result.

    Phase 1 (add): compute a+b into result cell (Specialist A pattern)
    Phase 2 (negate): negate the result (Specialist A pattern, but combined)

    Data layout: 24=a, 25=b(also initial result), 26=final_result, 27=temp
    """
    a_val, b_val = clamp(a), clamp(b)
    assert abs(a_val + b_val) <= VALUE_MAX, f"Overflow: {a_val}+{b_val}"

    mem = [0] * MEM_SIZE

    # Phase 1: Add — result starts as b, subtract (-a) to get b+a
    # Instr 0 (pc=0): clear temp
    mem[0] = 27; mem[1] = 27; mem[2] = 3
    # Instr 1 (pc=3): temp -= a → temp = -a
    mem[3] = 24; mem[4] = 27; mem[5] = 6
    # Instr 2 (pc=6): result -= temp → result -= (-a) = result + a = b + a
    mem[6] = 27; mem[7] = 25; mem[8] = 9

    # Phase 2: Negate — put -(a+b) into final_result
    # Instr 3 (pc=9): clear final_result
    mem[9] = 26; mem[10] = 26; mem[11] = 12
    # Instr 4 (pc=12): final_result -= sum → final_result = -(a+b)
    mem[12] = 25; mem[13] = 26; mem[14] = 15

    # Instr 5 (pc=15): halt
    mem[15] = 27; mem[16] = 27; mem[17] = -1

    # Data
    mem[24] = a_val    # a
    mem[25] = b_val    # b (also accumulator for sum)
    mem[26] = 0        # final result (will hold -(a+b))
    mem[27] = 0        # temp

    expected = clamp(-(a_val + b_val))
    return mem, 0, 26, expected


def make_multiply_then_negate(a, b):
    """Compute -(a*b): multiply then negate.

    Phase 1 (multiply): repeated addition loop (Specialist B)
    Phase 2 (negate): negate the product (Specialist A)

    Data layout: 24=-a, 25=counter(b), 26=product, 27=const1, 28=neg_result, 29=temp
    """
    a_val, b_val = abs(clamp(a)), abs(clamp(b))
    assert a_val * b_val <= VALUE_MAX, f"Overflow: {a_val}*{b_val}"

    mem = [0] * MEM_SIZE

    # Phase 1: Multiply loop
    # Instr 0 (pc=0): product -= (-a) → product += a
    mem[0] = 24; mem[1] = 26; mem[2] = 3
    # Instr 1 (pc=3): counter -= 1, if <=0 goto negate phase (pc=9)
    mem[3] = 27; mem[4] = 25; mem[5] = 9
    # Instr 2 (pc=6): trampoline to 0
    mem[6] = 29; mem[7] = 29; mem[8] = 0

    # Phase 2: Negate product
    # Instr 3 (pc=9): clear neg_result
    mem[9] = 28; mem[10] = 28; mem[11] = 12
    # Instr 4 (pc=12): neg_result -= product → neg_result = -product
    mem[12] = 26; mem[13] = 28; mem[14] = 15

    # Instr 5 (pc=15): halt
    mem[15] = 29; mem[16] = 29; mem[17] = -1

    # Data
    mem[24] = clamp(-a_val)  # -a
    mem[25] = b_val           # counter
    mem[26] = 0               # product
    mem[27] = 1               # constant 1
    mem[28] = 0               # neg_result (final answer)
    mem[29] = 0               # temp

    expected = clamp(-(a_val * b_val))
    return mem, 0, 28, expected
