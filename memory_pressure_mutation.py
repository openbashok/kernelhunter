"""
Module: memory_pressure_mutation.py

Description:
Generates memory pressure mutations for KernelHunter shellcodes.
Creates aggressive memory operations (mmap, munmap, mprotect) to stress the kernel's memory
management subsystems, increasing the chance of finding heap corruptions, race conditions,
or fragmentation-related vulnerabilities.
"""

import random

# Syscall numbers for memory operations (Linux x86_64)
SYSCALL_MMAP = 9
SYSCALL_MUNMAP = 11
SYSCALL_MPROTECT = 10

# Constants
SYSCALL_INSTR = b"\x0f\x05"  # syscall instruction


def generate_memory_pressure_fragment(min_ops=3, max_ops=10):
    """
    Generates a sequence of memory-related syscalls.

    Args:
        min_ops (int): Minimum number of operations.
        max_ops (int): Maximum number of operations.

    Returns:
        bytes: Shellcode fragment with memory pressure operations.
    """
    num_ops = random.randint(min_ops, max_ops)
    fragment = b""

    for _ in range(num_ops):
        syscall_choice = random.choice([SYSCALL_MMAP, SYSCALL_MUNMAP, SYSCALL_MPROTECT])
        setup = b"\x48\xc7\xc0" + syscall_choice.to_bytes(4, byteorder='little')

        # Setup dummy arguments: clear rdi, rsi, rdx
        args_setup = b"\x48\x31\xff\x48\x31\xf6\x48\x31\xd2"

        fragment += setup + args_setup + SYSCALL_INSTR

    return fragment


# Example usage
if __name__ == "__main__":
    fragment = generate_memory_pressure_fragment()
    print(f"Generated memory pressure fragment: {fragment.hex()}")
