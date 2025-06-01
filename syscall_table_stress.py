"""
Module: syscall_table_stress.py

Description:
Generates fragments targeting syscall table stress for KernelHunter shellcodes.
Performs rapid sequences of invalid or borderline syscalls with random arguments,
attempting to saturate the kernel's syscall handling mechanisms, descriptor tables,
or internal validation routines.

Probable Impact:
- Descriptor table corruption
- Undefined behavior from invalid syscall arguments
- Kernel panic from handling unexpected syscall states

Risk Level:
High - Can cause instability in syscall validation logic and panic if mishandled.
"""

import random

# Common and extended syscall numbers (Linux x86_64)
VALID_SYSCALLS = [
    0, 1, 2, 3, 9, 10, 11, 12, 39, 41, 60, 231
]

INVALID_SYSCALLS = [
    400, 500, 600, 700, 800, 999, 1000  # Out of range or undefined
]

# Constants
SYSCALL_INSTR = b"\x0f\x05"


def generate_syscall_table_stress_fragment(min_calls=5, max_calls=20):
    """
    Generates a fragment stressing the syscall handling subsystem.

    Args:
        min_calls (int): Minimum number of syscall attempts.
        max_calls (int): Maximum number of syscall attempts.

    Returns:
        bytes: Shellcode fragment for syscall table stress.
    """
    num_calls = random.randint(min_calls, max_calls)
    fragment = b""

    for _ in range(num_calls):
        if random.random() < 0.5:
            syscall_num = random.choice(VALID_SYSCALLS)
        else:
            syscall_num = random.choice(INVALID_SYSCALLS)

        setup = b"\x48\xc7\xc0" + syscall_num.to_bytes(4, byteorder='little')

        # Random arguments setup (can be garbage)
        args_setup = random.choice([
            b"\x48\x31\xff",  # xor rdi, rdi
            b"\x48\x31\xf6",  # xor rsi, rsi
            b"\x48\x31\xd2",  # xor rdx, rdx
            b"\x90"             # NOP
        ])

        fragment += setup + args_setup + SYSCALL_INSTR

    return fragment


# Example usage
if __name__ == "__main__":
    fragment = generate_syscall_table_stress_fragment()
    print(f"Generated syscall table stress fragment: {fragment.hex()}")
