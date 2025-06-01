"""
Module: extract_function_adn.py

Description:
Extracts complete functions from ELF binaries to use as intelligent genetic material (ADN)
for KernelHunter. Falls back to extracting random .text section fragments if no symbols are available.
Allows configuration of minimum and maximum function size for extraction.
"""

import os
import random
from elftools.elf.elffile import ELFFile

# List of binaries to scan for functions
BINARIES_TO_SCAN = [
    "/bin/ls",
    "/bin/cat",
    "/usr/bin/id",
    "/usr/bin/ssh",
    "/usr/bin/top",
    "/usr/bin/bash",
]

# Minimum and maximum fragment size fallback
FRAGMENT_MIN_LEN = 8
FRAGMENT_MAX_LEN = 24

# Configurable size limits for extracted functions
MIN_FUNCTION_SIZE = 16
MAX_FUNCTION_SIZE = 128

def extract_function_from_elf(filepath):
    try:
        with open(filepath, 'rb') as f:
            elf = ELFFile(f)
            symtab = elf.get_section_by_name('.symtab')
            if not symtab:
                return None  # No symbols available

            func_symbols = [sym for sym in symtab.iter_symbols()
                            if sym['st_info']['type'] == 'STT_FUNC'
                            and MIN_FUNCTION_SIZE <= sym['st_size'] <= MAX_FUNCTION_SIZE]
            if not func_symbols:
                return None

            selected_func = random.choice(func_symbols)
            addr = selected_func['st_value']
            size = selected_func['st_size']

            # Locate .text section
            text_section = elf.get_section_by_name('.text')
            if not text_section:
                return None

            # Calculate offset inside .text
            section_start = text_section['sh_addr']
            offset = addr - section_start
            if offset < 0 or offset + size > text_section.data_size:
                return None

            return text_section.data()[offset:offset + size]
    except Exception as e:
        print(f"[ERROR] Failed to extract function from {filepath}: {e}")
        return None

def fallback_extract_text_fragment(filepath):
    try:
        with open(filepath, 'rb') as f:
            elf = ELFFile(f)
            text_section = elf.get_section_by_name('.text')
            if not text_section:
                return b"\x90"

            data = text_section.data()
            if len(data) < FRAGMENT_MAX_LEN:
                return b"\x90"

            start = random.randint(0, len(data) - FRAGMENT_MAX_LEN)
            length = random.randint(FRAGMENT_MIN_LEN, FRAGMENT_MAX_LEN)
            return data[start:start + length]
    except Exception as e:
        print(f"[ERROR] Fallback extraction failed for {filepath}: {e}")
        return b"\x90"

def get_function_or_fragment():
    bin_path = random.choice(BINARIES_TO_SCAN)

    func_fragment = extract_function_from_elf(bin_path)
    if func_fragment:
        return func_fragment

    # Fallback if no function was extracted
    return fallback_extract_text_fragment(bin_path)

# Example usage
if __name__ == "__main__":
    fragment = get_function_or_fragment()
    print(f"Extracted ADN fragment: {fragment.hex()}")
