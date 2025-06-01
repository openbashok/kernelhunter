"""
Module: speculative_confusion.py

Description:
Generates speculative execution noise for KernelHunter shellcodes.
Inserts instructions related to memory fences, cache flushing, and speculation control,
with the goal of destabilizing speculative execution paths inside the kernel.
This can expose subtle timing-related bugs and side-channel vulnerabilities.
"""

import random

def generate_speculative_confusion_fragment(min_instr=3, max_instr=8):
    """
    Generates a fragment with speculative execution disruption instructions.

    Args:
        min_instr (int): Minimum number of instructions.
        max_instr (int): Maximum number of instructions.

    Returns:
        bytes: Shellcode fragment with speculative confusion patterns.
    """
    num_instr = random.randint(min_instr, max_instr)
    fragment = b""

    speculative_instructions = [
        b"\x0f\xae\xf0",  # MFENCE
        b"\x0f\xae\xe8",  # LFENCE
        b"\x0f\xae\xf8",  # SFENCE
        b"\x0f\xae\x38",  # CLFLUSH [rax]
        b"\x0f\x18\x08",  # PREFETCHNTA [rax]
        b"\x0f\x18\x09",  # PREFETCHT0 [rcx]
    ]

    for _ in range(num_instr):
        instr = random.choice(speculative_instructions)
        fragment += instr

    return fragment

# Example usage
if __name__ == "__main__":
    fragment = generate_speculative_confusion_fragment()
    print(f"Generated speculative confusion fragment: {fragment.hex()}")
