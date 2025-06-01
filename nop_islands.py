"""
Module: nop_islands.py

Description:
This module generates small fragments composed of NOP instructions ("NOP islands")
to facilitate safer recombination during genetic evolution of shellcodes.
NOP islands act as natural cut-and-paste points during crossover operations, allowing
mutations or insertions without breaking instruction flow or corrupting execution.

Supports different sizes of NOPs to introduce slight randomness while remaining safe.
Includes internal control to prevent overpopulation of NOPs during evolution.
"""

import random

# NOP instruction patterns of different lengths
NOP_PATTERNS = [
    b"\x90",            # 1-byte NOP
    b"\x66\x90",        # 2-byte NOP
    b"\x0f\x1f\x00",    # 3-byte NOP
    b"\x0f\x1f\x40\x00" # 4-byte NOP
]

# Internal counter to avoid flooding with NOPs
NOP_INSERTION_COUNTER = 0
MAX_NOP_INSERTIONS_PER_GEN = 50  # adjust depending on evolution size


def generate_nop_island(min_nops=2, max_nops=8):
    """
    Generate a small island of NOP instructions.

    Args:
        min_nops (int): minimum number of NOP instructions.
        max_nops (int): maximum number of NOP instructions.

    Returns:
        bytes: sequence of NOP instructions.
    """
    global NOP_INSERTION_COUNTER

    if NOP_INSERTION_COUNTER >= MAX_NOP_INSERTIONS_PER_GEN:
        # Too many NOPs already inserted, fallback to minimal NOP
        return b"\x90"

    num_nops = random.randint(min_nops, max_nops)
    nop_island = b""

    for _ in range(num_nops):
        nop = random.choice(NOP_PATTERNS)
        nop_island += nop

    NOP_INSERTION_COUNTER += num_nops

    return nop_island


def reset_nop_counter():
    """Reset NOP insertion counter (should be called at the start of each generation)."""
    global NOP_INSERTION_COUNTER
    NOP_INSERTION_COUNTER = 0


# Example usage
if __name__ == "__main__":
    reset_nop_counter()
    fragment = generate_nop_island()
    print(f"Generated NOP island: {fragment.hex()}")
