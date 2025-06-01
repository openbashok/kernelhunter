"""
Module: control_flow_traps.py

Description:
Generates control flow disruption traps for KernelHunter shellcodes.
Inserts random jumps, calls, and returns to destabilize normal execution paths,
potentially causing unexpected control transfers, stack corruption, or illegal instruction
exceptions inside the kernel.
"""

import random

def generate_control_flow_trap_fragment(min_instr=2, max_instr=6):
    """
    Generates a fragment with random control flow disruption instructions.

    Args:
        min_instr (int): Minimum number of control flow instructions.
        max_instr (int): Maximum number of control flow instructions.

    Returns:
        bytes: Shellcode fragment with control flow traps.
    """
    num_instr = random.randint(min_instr, max_instr)
    fragment = b""

    control_flow_instructions = [
        b"\xeb\x00",          # JMP short (0 offset)
        b"\xe8\x00\x00\x00\x00",  # CALL next instruction
        b"\xc3",              # RET
        b"\xc2\x00\x00",     # RET with stack adjustment
        b"\x0f\x05",          # SYSCALL (acts as an implicit control flow change)
    ]

    for _ in range(num_instr):
        instr = random.choice(control_flow_instructions)
        fragment += instr

    return fragment

# Example usage
if __name__ == "__main__":
    fragment = generate_control_flow_trap_fragment()
    print(f"Generated control flow trap fragment: {fragment.hex()}")
