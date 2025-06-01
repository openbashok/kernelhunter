"""
Module: syscall_storm.py

Description:
Generates aggressive sequences of syscall setups and invocations for KernelHunter shellcodes.
Designed to overwhelm the kernel with multiple rapid syscalls, increasing the chance
of triggering race conditions, resource exhaustion, or critical failures under fuzzing stress.
"""

import random

# List of interesting syscall numbers (Linux x86_64)
INTERESTING_SYSCALLS = [
    0,    # read
    1,    # write
    2,    # open
    3,    # close
    9,    # mmap
    10,   # mprotect
    11,   # munmap
    12,   # brk
    56,   # clone
    57,   # fork
    58,   # vfork
    59,   # execve
    60,   # exit
    61,   # wait4
    231,  # exit_group
]

# Constants
SYSCALL_INSTR = b"\x0f\x05"  # syscall instruction


def generate_syscall_storm(min_calls=5, max_calls=20):
    """
    Generate a sequence of syscall setups and invocations.

    Args:
        min_calls (int): Minimum number of syscalls.
        max_calls (int): Maximum number of syscalls.

    Returns:
        bytes: Shellcode sequence with multiple syscalls.
    """
    num_calls = random.randint(min_calls, max_calls)
    storm = b""

    for _ in range(num_calls):
        syscall_num = random.choice(INTERESTING_SYSCALLS)
        setup = b"\x48\xc7\xc0" + syscall_num.to_bytes(4, byteorder='little')
        # Optionally randomize arguments slightly (zeroing rdi, rsi)
        args_setup = random.choice([
            b"\x48\x31\xff",  # xor rdi, rdi
            b"\x48\x31\xf6",  # xor rsi, rsi
            b"\x48\x31\xd2",  # xor rdx, rdx
            b"\x90"             # NOP (do nothing)
        ])
        storm += setup + args_setup + SYSCALL_INSTR

    return storm


# Example usage
if __name__ == "__main__":
    fragment = generate_syscall_storm()
    print(f"Generated syscall storm fragment: {fragment.hex()}")
