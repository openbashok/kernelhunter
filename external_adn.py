import os
import random
from elftools.elf.elffile import ELFFile

# Lista de binarios del sistema de los que extraer fragmentos
BINARIES_TO_SCAN = [
    "/bin/ls",
    "/bin/cat",
    "/usr/bin/id",
    "/usr/bin/ssh",
    "/usr/bin/top",
    "/usr/bin/bash",
]

FRAGMENT_MIN_LEN = 8
FRAGMENT_MAX_LEN = 24

POTENTIAL_START_BYTES = [
    0x48, 0x55, 0x53, 0x57, 0x41, 0x50, 0x90
]

def extract_text_section_bytes(filepath):
    try:
        with open(filepath, 'rb') as f:
            elf = ELFFile(f)
            text_section = elf.get_section_by_name('.text')
            if not text_section:
                raise Exception(f"No .text section in {filepath}")
            return text_section.data()
    except Exception as e:
        print(f"[ERROR] ELF parsing failed for {filepath}: {e}")
        return b""

def get_random_fragment_from_bin():
    bin_path = random.choice(BINARIES_TO_SCAN)
    data = extract_text_section_bytes(bin_path)

    if len(data) < FRAGMENT_MAX_LEN:
        return b"\x90"

    for _ in range(20):
        start = random.randint(0, len(data) - FRAGMENT_MAX_LEN)
        length = random.randint(FRAGMENT_MIN_LEN, FRAGMENT_MAX_LEN)
        fragment = data[start:start + length]

        if fragment.count(b'\x00') > 3:
            continue

        if fragment[0] not in POTENTIAL_START_BYTES:
            continue

        return fragment

    return b"\x90"

# Integración en generate_random_instruction()
# Agregá esto como nueva opción:
# elif choice_type == "external_adn":
#     return get_random_fragment_from_bin()
