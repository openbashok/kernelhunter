"""
Module: gene_bank_advanced_v2.py

Description:
Advanced dynamic Gene Bank for KernelHunter shellcodes.
Provides functional genes capable of generating system calls with randomized safe parameters.
Focuses on I/O, memory operations, and process control (no networking yet).

Future Implementations:
- Separate networking genes into a dedicated Networking Gene Bank module.
- Introduce signaling and timer-related genes.
- Implement smarter adaptive mutations on gene parameters.

Probable Impact:
- High - Enables realistic system interaction, triggers resource allocation, and stresses kernel pathways.

Risk Level:
⚡⚡ Moderate to High - Can trigger real system activity, resource exhaustion, and potential instabilities.
"""

import random

# Gene generators

def gene_write(fd=None, buf_addr=None, length=None):
    if fd is None:
        fd = random.randint(0, 5)  # stdout, stderr, small fds
    if buf_addr is None:
        buf_addr = random.randint(0x1000, 0x100000)
    if length is None:
        length = random.randint(4, 4096)
    return (
        b"\x48\xc7\xc0\x01\x00\x00\x00"  # sys_write
        + b"\x48\xc7\xc7" + fd.to_bytes(4, 'little')
        + b"\x48\xc7\xc6" + buf_addr.to_bytes(4, 'little')
        + b"\x48\xc7\xc2" + length.to_bytes(4, 'little')
        + b"\x0f\x05"
    )

def gene_read(fd=None, buf_addr=None, length=None):
    if fd is None:
        fd = random.randint(0, 5)
    if buf_addr is None:
        buf_addr = random.randint(0x1000, 0x100000)
    if length is None:
        length = random.randint(4, 4096)
    return (
        b"\x48\xc7\xc0\x00\x00\x00\x00"  # sys_read
        + b"\x48\xc7\xc7" + fd.to_bytes(4, 'little')
        + b"\x48\xc7\xc6" + buf_addr.to_bytes(4, 'little')
        + b"\x48\xc7\xc2" + length.to_bytes(4, 'little')
        + b"\x0f\x05"
    )

def gene_open(path_addr=None, flags=0, mode=0):
    if path_addr is None:
        path_addr = random.randint(0x1000, 0x20000)
    return (
        b"\x48\xc7\xc0\x02\x00\x00\x00"  # sys_open
        + b"\x48\xc7\xc7" + path_addr.to_bytes(4, 'little')
        + b"\x48\xc7\xc6" + flags.to_bytes(4, 'little')
        + b"\x48\xc7\xc2" + mode.to_bytes(4, 'little')
        + b"\x0f\x05"
    )

def gene_close(fd=None):
    if fd is None:
        fd = random.randint(0, 1024)
    return (
        b"\x48\xc7\xc0\x03\x00\x00\x00"  # sys_close
        + b"\x48\xc7\xc7" + fd.to_bytes(4, 'little')
        + b"\x0f\x05"
    )

def gene_mmap(addr=None, length=None, prot=0x7, flags=0x22):
    if addr is None:
        addr = 0
    if length is None:
        length = random.randint(4096, 65536)
    return (
        b"\x48\xc7\xc0\x09\x00\x00\x00"  # sys_mmap
        + b"\x48\xc7\xc7" + addr.to_bytes(4, 'little')
        + b"\x48\xc7\xc6" + length.to_bytes(4, 'little')
        + b"\x48\xc7\xc2" + prot.to_bytes(4, 'little')
        + b"\x0f\x05"
    )

def gene_munmap(addr=None, length=None):
    if addr is None:
        addr = random.randint(0x1000, 0x100000)
    if length is None:
        length = random.randint(4096, 65536)
    return (
        b"\x48\xc7\xc0\x0b\x00\x00\x00"  # sys_munmap
        + b"\x48\xc7\xc7" + addr.to_bytes(4, 'little')
        + b"\x48\xc7\xc6" + length.to_bytes(4, 'little')
        + b"\x0f\x05"
    )

def gene_fork():
    return b"\x48\xc7\xc0\x39\x00\x00\x00\x0f\x05"  # sys_fork

def gene_clone(flags=0x00000100):
    return (
        b"\x48\xc7\xc0\x38\x00\x00\x00"  # sys_clone
        + b"\x48\xc7\xc7" + flags.to_bytes(4, 'little')
        + b"\x0f\x05"
    )

def gene_execve(path_addr=None, argv_addr=0, envp_addr=0):
    if path_addr is None:
        path_addr = random.randint(0x1000, 0x20000)
    return (
        b"\x48\xc7\xc0\x3b\x00\x00\x00"  # sys_execve
        + b"\x48\xc7\xc7" + path_addr.to_bytes(4, 'little')
        + b"\x48\xc7\xc6" + argv_addr.to_bytes(4, 'little')
        + b"\x48\xc7\xc2" + envp_addr.to_bytes(4, 'little')
        + b"\x0f\x05"
    )

# Gene dispatcher
GENE_FUNCTIONS = [
    gene_write,
    gene_read,
    gene_open,
    gene_close,
    gene_mmap,
    gene_munmap,
    gene_fork,
    gene_clone,
    gene_execve,
]

def get_random_gene_dynamic():
    """
    Selects and executes a random dynamic gene function.

    Returns:
        bytes: A shellcode fragment representing a dynamic gene.
    """
    func = random.choice(GENE_FUNCTIONS)
    return func()

def list_dynamic_genes():
    """
    Lists all available dynamic gene function names.

    Returns:
        list: Gene function names.
    """
    return [func.__name__ for func in GENE_FUNCTIONS]

# Example usage
if __name__ == "__main__":
    print("Available dynamic genes:", list_dynamic_genes())
    fragment = get_random_gene_dynamic()
    print(f"Generated dynamic gene fragment: {fragment.hex()}")
