"""
Module: interrupt_storm.py

Description:
Generates rapid bursts of interrupts (software and hardware IRQs),
attempting to trigger race conditions and synchronization issues
within the kernel's interrupt handlers.

Expected Impact:
- Race conditions in IRQ handlers.
- Kernel synchronization issues leading to potential instability or crashes.

Risk Level:
High
"""

import random

def generate_interrupt_storm_fragment(min_interrupts=5, max_interrupts=15):
    """
    Generates a shellcode fragment causing rapid software interrupts.

    Args:
        min_interrupts (int): Minimum number of interrupts.
        max_interrupts (int): Maximum number of interrupts.

    Returns:
        bytes: Shellcode fragment with multiple interrupt instructions.
    """
    num_interrupts = random.randint(min_interrupts, max_interrupts)
    fragment = b""

    # Interrupt instructions (int3 for debugging, int 0x80 for legacy syscalls)
    interrupts = [
        b"\xcc",      # INT 3 (breakpoint interrupt)
        b"\xcd\x80",  # INT 0x80 (legacy syscall interrupt)
        b"\xce"       # INTO (Overflow interrupt)
    ]

    for _ in range(num_interrupts):
        fragment += random.choice(interrupts)

    return fragment

# Ejemplo de uso rápido
if __name__ == "__main__":
    fragment = generate_interrupt_storm_fragment()
    print(f"Interrupt storm shellcode: {fragment.hex()}")
