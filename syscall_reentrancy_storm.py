"""
Module: syscall_reentrancy_storm.py

Description:
Generates syscall reentrancy storm fragments for KernelHunter shellcodes.
Aggressively sequences syscalls like clone, fork, execve within shellcodes
to simulate reentrant or nested syscall execution scenarios, targeting race
conditions, lock corruption, and kernel synchronization vulnerabilities.

Probable Impact:
- Race conditions inside syscall handlers
- Lock corruption or double unlock situations
- Kernel panic due to inconsistent process states

Risk Level:
Very High - Can induce severe race conditions and kernel instability under stress.
"""

import random

# Syscall numbers for process-related syscalls (Linux x86_64)
SYSCALL_CLONE = 56
SYSCALL_FORK = 57
SYSCALL_EXECVE = 59
SYSCALL_EXIT = 60
SYSCALL_EXIT_GROUP = 231

# Constants
SYSCALL_INSTR = b"\x0f\x05"


def generate_syscall_reentrancy_storm_fragment(min_chains=3, max_chains=10):
    """
    Generates a fragment creating syscall reentrancy storms.

    Args:
        min_chains (int): Minimum number of nested syscall chains.
        max_chains (int): Maximum number of nested syscall chains.

    Returns:
        bytes: Shellcode fragment with syscall reentrancy behavior.
    """
    num_chains = random.randint(min_chains, max_chains)
    fragment = b""

    reentrant_syscalls = [
        SYSCALL_CLONE,
        SYSCALL_FORK,
        SYSCALL_EXECVE,
        SYSCALL_EXIT,
        SYSCALL_EXIT_GROUP,
    ]

    for _ in range(num_chains):
        syscall_choice = random.choice(reentrant_syscalls)
        setup = b"\x48\xc7\xc0" + syscall_choice.to_bytes(4, byteorder='little')

        args_setup = b"\x48\x31\xff\x48\x31\xf6\x48\x31\xd2"

        fragment += setup + args_setup + SYSCALL_INSTR

    return fragment


# Example usage
if __name__ == "__main__":
    fragment = generate_syscall_reentrancy_storm_fragment()
    print(f"Generated syscall reentrancy storm fragment: {fragment.hex()}")
