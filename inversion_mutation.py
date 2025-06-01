"""
Module: inversion_mutation.py

Description:
Implements genetic inversion mutation for KernelHunter shellcodes.
Randomly selects a fragment of the shellcode and reverses its order.
Inversion can create new execution paths and disrupt known sequences
without destroying overall structure, promoting deeper evolutionary exploration.
"""

import random

def invert_fragment(shellcode, min_size=4, max_size=16):
    """
    Randomly selects a fragment of the shellcode and reverses its byte order.

    Args:
        shellcode (bytes): The original shellcode.
        min_size (int): Minimum size of the fragment to invert.
        max_size (int): Maximum size of the fragment to invert.

    Returns:
        bytes: Mutated shellcode with an inverted fragment.
    """
    if len(shellcode) < min_size + 1:
        return shellcode

    frag_len = random.randint(min_size, min(max_size, len(shellcode)))
    start_idx = random.randint(0, len(shellcode) - frag_len)
    fragment = shellcode[start_idx:start_idx + frag_len]

    inverted_fragment = fragment[::-1]  # Reverse the fragment

    new_shellcode = shellcode[:start_idx] + inverted_fragment + shellcode[start_idx + frag_len:]

    return new_shellcode

# Example usage
if __name__ == "__main__":
    original = b"\x48\x31\xc0\x48\x89\xc2\xb0\x01\x0f\x05"
    mutated = invert_fragment(original)
    print(f"Original: {original.hex()}")
    print(f"Mutated : {mutated.hex()}")
