"""
Module: neutral_mutation.py

Description:
Implements silent neutral mutations for KernelHunter shellcodes.
Randomly inserts small neutral instructions (e.g., NOPs, harmless register operations)
that do not affect the functional behavior immediately. This technique helps to explore
neutral paths in the evolutionary space, enabling deeper and more robust shellcode evolution.
"""

import random

def insert_neutral_mutation(shellcode, min_insertions=1, max_insertions=3):
    """
    Randomly inserts neutral instructions into the shellcode.

    Args:
        shellcode (bytes): The original shellcode.
        min_insertions (int): Minimum number of neutral mutations.
        max_insertions (int): Maximum number of neutral mutations.

    Returns:
        bytes: Mutated shellcode with neutral instructions inserted.
    """
    neutral_instructions = [
        b'\x90',             # NOP
        b'\x66\x90',         # 2-byte NOP
        b'\x31\xc0',         # XOR EAX, EAX (harmless reset)
        b'\x48\x89\c0',     # MOV RAX, RAX (noop)
        b'\x48\x89\c9',     # MOV RCX, RCX (noop)
    ]

    new_shellcode = bytearray(shellcode)
    num_insertions = random.randint(min_insertions, max_insertions)

    for _ in range(num_insertions):
        insert_idx = random.randint(0, len(new_shellcode))
        neutral = random.choice(neutral_instructions)
        new_shellcode = new_shellcode[:insert_idx] + neutral + new_shellcode[insert_idx:]

    return bytes(new_shellcode)

# Example usage
if __name__ == "__main__":
    original = b"\x48\x31\xc0\x48\x89\xc2\xb0\x01\x0f\x05"
    mutated = insert_neutral_mutation(original)
    print(f"Original: {original.hex()}")
    print(f"Mutated : {mutated.hex()}")
