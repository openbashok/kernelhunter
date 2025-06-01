"""
Module: duplication_mutation.py

Description:
Implements genetic massive duplication for KernelHunter shellcodes.
Randomly selects a fragment of the shellcode and duplicates it one or more times
at random insertion points. Duplication increases the chance of reinforcing
useful instruction sequences and creating stronger evolutionary traits.
"""

import random

def duplicate_fragment(shellcode, min_size=4, max_size=16, min_copies=2, max_copies=5):
    """
    Randomly selects a fragment of the shellcode and duplicates it multiple times.

    Args:
        shellcode (bytes): The original shellcode.
        min_size (int): Minimum size of the fragment to duplicate.
        max_size (int): Maximum size of the fragment to duplicate.
        min_copies (int): Minimum number of times to duplicate.
        max_copies (int): Maximum number of times to duplicate.

    Returns:
        bytes: Mutated shellcode with fragment duplications.
    """
    if len(shellcode) < min_size:
        return shellcode

    frag_len = random.randint(min_size, min(max_size, len(shellcode)))
    start_idx = random.randint(0, len(shellcode) - frag_len)
    fragment = shellcode[start_idx:start_idx + frag_len]

    num_copies = random.randint(min_copies, max_copies)
    new_shellcode = bytearray(shellcode)

    for _ in range(num_copies):
        insert_idx = random.randint(0, len(new_shellcode))
        new_shellcode = new_shellcode[:insert_idx] + fragment + new_shellcode[insert_idx:]

    return bytes(new_shellcode)

# Example usage
if __name__ == "__main__":
    original = b"\x48\x31\xc0\x48\x89\xc2\xb0\x01\x0f\x05"
    mutated = duplicate_fragment(original)
    print(f"Original: {original.hex()}")
    print(f"Mutated : {mutated.hex()}")
