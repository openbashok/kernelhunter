import random

# NOP patterns we recognize for safe cuts
NOP_PATTERNS = [
    b"\x90",
    b"\x66\x90",
    b"\x0f\x1f\x00",
    b"\x0f\x1f\x40\x00"
]

def find_nop_cut(shellcode):
    """Attempts to find a NOP island to cut at. Falls back to random cut if none found."""
    for i in range(len(shellcode)):
        for nop in NOP_PATTERNS:
            if shellcode[i:i+len(nop)] == nop:
                return i + len(nop)  # cut after the NOP
    # fallback
    return safe_cut(shellcode)

def safe_cut(shellcode, min_cut=4, max_cut=16):
    """Finds a safe cut point in shellcode (fallback random cut)."""
    if len(shellcode) <= min_cut:
        return 1
    return random.randint(min_cut, min(len(shellcode) - 1, max_cut))

def insert_glue_instruction():
    """Returns a random small glue instruction."""
    glues = [
        b"\x90",               # NOP
        b"\x48\x31\xc0",        # xor rax, rax
        b"\xc3",                # ret
        b"\x50\x58",            # push rax; pop rax
        b"\x31\xc0",            # xor eax, eax
    ]
    return random.choice(glues)

def mutate_edges(shellcode):
    """Applies a slight mutation at the beginning or end of the shellcode."""
    if len(shellcode) < 6:
        return shellcode
    mutate_pos = random.choice(["start", "end"])
    mutation = bytes([random.randint(0x00, 0xFF) for _ in range(3)])
    if mutate_pos == "start":
        return mutation + shellcode[3:]
    else:
        return shellcode[:-3] + mutation

def crossover_shellcodes_advanced(parent_a, parent_b):
    """Advanced crossover combining two shellcodes, preferring cuts at NOP islands."""
    if not parent_a or not parent_b:
        return b""

    if random.random() < 0.5:
        # Single crossover
        cut_a = find_nop_cut(parent_a)
        cut_b = find_nop_cut(parent_b)
        part_a = parent_a[:cut_a]
        part_b = parent_b[cut_b:]
        child = part_a + insert_glue_instruction() + part_b
    else:
        # Double crossover (triple recombination)
        cut_a1 = find_nop_cut(parent_a)
        cut_a2 = find_nop_cut(parent_a[cut_a1:]) + cut_a1
        cut_b = find_nop_cut(parent_b)
        part1 = parent_a[:cut_a1]
        part2 = parent_b[:cut_b]
        part3 = parent_a[cut_a2:]
        child = part1 + insert_glue_instruction() + part2 + insert_glue_instruction() + part3

    # Optionally mutate edges
    if random.random() < 0.3:
        child = mutate_edges(child)

    return child

# Example usage
if __name__ == "__main__":
    a = b"\x48\x31\xc0\x90\x48\x31\xff\x0f\x05"
    b = b"\x48\x89\xe7\x66\x90\xb0\x3b\x0f\x05"
    child = crossover_shellcodes_advanced(a, b)
    print(f"Parent A: {a.hex()}")
    print(f"Parent B: {b.hex()}")
    print(f"Child   : {child.hex()}")