"""
Module: gene_bank.py

Description:
Defines a Gene Bank for KernelHunter shellcodes.
Provides prebuilt functional shellcode fragments ("genes") that perform basic system operations
like writing to stdout, socket operations, memory management, and process manipulation.
These genes act as foundational building blocks for more complex evolved behaviors during fuzzing.

Future Implementations:
- Extend genes into dynamic generators capable of accepting parameters (e.g., file descriptors, buffer addresses).
- Introduce modular genes for more complex operations (e.g., full TCP handshake simulation).
- Enable genetic memory to prioritize historically effective genes during evolution.

Probable Impact:
- Increased interaction with kernel I/O, networking, and process subsystems.
- Expanded kernel code coverage and potential vulnerability surface.

Risk Level:
⚡ Moderate to High - Enables active environmental interaction, potential resource exhaustion, and kernel state destabilization.
"""

import random

# Prebuilt functional genes
GENE_BANK = {
    "gene_write_stdout": b"\x48\xc7\xc0\x01\x00\x00\x00"  # mov rax, 1 (sys_write)
                         b"\x48\x31\xff"                     # xor rdi, rdi (stdout)
                         b"\x48\x31\xf6"                     # xor rsi, rsi (NULL buffer)
                         b"\x48\x31\xd2"                     # xor rdx, rdx (length=0)
                         b"\x0f\x05",

    "gene_socket_tcp": b"\x48\xc7\xc0\x29\x00\x00\x00"    # mov rax, 41 (sys_socket)
                        b"\x48\x31\xff"                      # xor rdi, rdi (AF_INET)
                        b"\x48\x31\xf6"                      # xor rsi, rsi (SOCK_STREAM)
                        b"\x48\x31\xd2"                      # xor rdx, rdx (protocol=0)
                        b"\x0f\x05",

    "gene_send_stdout": b"\x48\xc7\xc0\x01\x00\x00\x00"
                         b"\x48\x31\xff"
                         b"\x48\x8d\x35\x04\x00\x00\x00"
                         b"\xba\x05\x00\x00\x00"
                         b"\x0f\x05"
                         b"hello",

    "gene_open_file": b"\x48\xc7\xc0\x02\x00\x00\x00"  # sys_open
                       b"\x48\x31\xff"                     # rdi (filename ptr, NULL)
                       b"\x48\x31\xf6"                     # rsi (flags, O_RDONLY)
                       b"\x48\x31\xd2"                     # rdx (mode)
                       b"\x0f\x05",

    "gene_read_file": b"\x48\xc7\xc0\x00\x00\x00\x00"  # sys_read
                       b"\x48\x31\xff"
                       b"\x48\x31\xf6"
                       b"\x48\x31\xd2"
                       b"\x0f\x05",

    "gene_close_fd": b"\x48\xc7\xc0\x03\x00\x00\x00"  # sys_close
                      b"\x48\x31\xff"
                      b"\x0f\x05",

    "gene_mmap_memory": b"\x48\xc7\xc0\x09\x00\x00\x00"  # sys_mmap
                         b"\x48\x31\xff"
                         b"\x48\x31\xf6"
                         b"\x48\x31\xd2"
                         b"\x0f\x05",

    "gene_munmap_memory": b"\x48\xc7\xc0\x0b\x00\x00\x00"  # sys_munmap
                           b"\x48\x31\xff"
                           b"\x48\x31\xf6"
                           b"\x0f\x05",

    "gene_dup_fd": b"\x48\xc7\xc0\x21\x00\x00\x00"  # sys_dup
                    b"\x48\x31\xff"
                    b"\x0f\x05",

    "gene_dup2_fd": b"\x48\xc7\xc0\x21\x00\x00\x00"  # sys_dup2 (repeated syscall id for now)
                     b"\x48\x31\xff"
                     b"\x48\x31\xf6"
                     b"\x0f\x05",

    "gene_fork_process": b"\x48\xc7\xc0\x39\x00\x00\x00"  # sys_fork
                          b"\x0f\x05",

    "gene_clone_process": b"\x48\xc7\xc0\x38\x00\x00\x00"  # sys_clone
                           b"\x0f\x05",

    "gene_exit_process": b"\x48\xc7\xc0\x3c\x00\x00\x00"  # sys_exit
                          b"\x48\x31\xff"
                          b"\x0f\x05",

    "gene_getpid": b"\x48\xc7\xc0\x27\x00\x00\x00"  # sys_getpid
                    b"\x0f\x05",

    "gene_getppid": b"\x48\xc7\xc0\x64\x00\x00\x00"  # sys_getppid
                     b"\x0f\x05",

    "gene_alarm_signal": b"\x48\xc7\xc0\x25\x00\x00\x00"  # sys_alarm
                          b"\x48\x31\xff"
                          b"\x0f\x05",

    "gene_setuid_zero": b"\x48\xc7\xc0\x69\x00\x00\x00"  # sys_setuid
                         b"\x48\x31\xff"
                         b"\x0f\x05",

    "gene_kill_self": b"\x48\xc7\xc0\x3e\x00\x00\x00"  # sys_kill
                       b"\x48\x31\xff"
                       b"\x48\x31\xf6"
                       b"\x0f\x05",
}


def get_random_gene():
    """
    Selects a random gene from the gene bank.

    Returns:
        bytes: A shellcode fragment representing a gene.
    """
    return random.choice(list(GENE_BANK.values()))


def list_genes():
    """
    Lists all available gene names.

    Returns:
        list: Gene names.
    """
    return list(GENE_BANK.keys())


# Example usage
if __name__ == "__main__":
    print("Available genes:", list_genes())
    fragment = get_random_gene()
    print(f"Selected gene fragment: {fragment.hex()}")
