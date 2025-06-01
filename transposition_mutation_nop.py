# transposition_mutation_nop.py

import random

# Recognized NOP patterns
NOP_PATTERNS = [
    b'\x90',
    b'\x66\x90',
    b'\x0f\x1f\x00',
    b'\x0f\x1f\x40\x00'
]

def find_nop_positions(shellcode):
    """Find all start positions where known NOP patterns appear."""
    positions = []
    for i in range(len(shellcode)):
        for nop in NOP_PATTERNS:
            if shellcode[i:i+len(nop)] == nop:
                positions.append(i)
    return positions

def transpose_fragment_nop_aware(shellcode, min_size=3, max_size=12):
    """
    Performs a NOP-aware transposition mutation: moves a small fragment between NOP islands.

    Args:
        shellcode (bytes): Shellcode to mutate.
        min_size (int): Minimum fragment size.
        max_size (int): Maximum fragment size.

    Returns:
        bytes: Mutated shellcode.
    """
    if len(shellcode) < min_size + 2:
        return shellcode

    nop_positions = find_nop_positions(shellcode)

    if len(nop_positions) >= 2:
        # Try to select a fragment between two NOPs
        start_idx = random.choice(nop_positions)
        possible_ends = [p for p in nop_positions if p > start_idx + min_size]
        if possible_ends:
            end_idx = random.choice(possible_ends)
            fragment = shellcode[start_idx:end_idx]

            # Remove the fragment
            shellcode_without = shellcode[:start_idx] + shellcode[end_idx:]

            # Try to insert into another NOP
            insert_positions = find_nop_positions(shellcode_without)
            if insert_positions:
                insert_idx = random.choice(insert_positions)
                mutated = shellcode_without[:insert_idx] + fragment + shellcode_without[insert_idx:]
                return mutated

    # If no suitable NOPs found, fallback to random fragment move
    frag_len = random.randint(min_size, min(max_size, len(shellcode)//2))
    start_idx = random.randint(0, len(shellcode) - frag_len)
    fragment = shellcode[start_idx:start_idx + frag_len]
    shellcode_without = shellcode[:start_idx] + shellcode[start_idx + frag_len:]
    insert_idx = random.randint(0, len(shellcode_without))
    mutated = shellcode_without[:insert_idx] + fragment + shellcode_without[insert_idx:]

    return mutated

# Example usage
if __name__ == "__main__":
    original = b"\x90\x48\x31\xc0\x90\x48\x89\xc2\x0f\x1f\x00\xb0\x01\x0f\x05\x90"
    mutated = transpose_fragment_nop_aware(original)
    print(f"Original: {original.hex()}")
    print(f"Mutated : {mutated.hex()}")
