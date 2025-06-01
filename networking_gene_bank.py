"""
Module: networking_gene_bank.py

Description:
Networking Gene Bank for KernelHunter shellcodes.
Provides dynamic genes for networking operations: opening sockets, connecting,
sending, receiving, and exchanging shellcode payloads between cells (crossover-based evolution).
Now includes reconnaissance functions to scan local ports, network interfaces,
and detect open services for active environmental interaction.

Future Improvements:
- Enhance with multi-step TCP handshakes.
- Implement passive listeners evolving on connection.
- Develop complex signaling mechanisms.

Probable Impact:
- Very High - Enables inter-cell communication, dynamic evolution, reconnaissance of the environment, and complex multi-shellcode behaviors.

Risk Level:
🚨 Very High - Interaction with real network stack, potential for deep kernel exposure.
"""

import random
import socket
import psutil

# Utilities

def get_local_ip():
    """Obtain local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

# Communication Genes

def gene_open_socket():
    """Generate a sys_socket call."""
    return (
        b"\x48\xc7\xc0\x29\x00\x00\x00"  # sys_socket
        + b"\x48\x31\xff"  # AF_INET
        + b"\x48\x31\xf6"  # SOCK_STREAM
        + b"\x48\x31\xd2"  # protocol=0
        + b"\x0f\x05"
    )

def gene_connect(ip=None, port=None):
    """Generate a sys_connect attempt (simplified for fuzzing)."""
    if ip is None:
        ip = get_local_ip()
    if port is None:
        port = random.randint(1024, 65535)
    return (
        b"\x48\xc7\xc0\x2a\x00\x00\x00"  # sys_connect
        + b"\x48\x31\xff"
        + b"\x48\x31\xf6"
        + b"\x48\x31\xd2"
        + b"\x0f\x05"
    )

def gene_send_shellcode(fd=None, payload_size=128):
    """Generate a sys_sendto to transmit shellcode (simplified)."""
    if fd is None:
        fd = random.randint(3, 1024)
    return (
        b"\x48\xc7\xc0\x2c\x00\x00\x00"  # sys_sendto
        + b"\x48\xc7\xc7" + fd.to_bytes(4, 'little')
        + b"\x48\x31\xf6"
        + b"\x48\xc7\xc2" + payload_size.to_bytes(4, 'little')
        + b"\x0f\x05"
    )

def gene_recv_shellcode(fd=None, buffer_size=128):
    """Generate a sys_recvfrom to receive shellcode (simplified)."""
    if fd is None:
        fd = random.randint(3, 1024)
    return (
        b"\x48\xc7\xc0\x2d\x00\x00\x00"  # sys_recvfrom
        + b"\x48\xc7\xc7" + fd.to_bytes(4, 'little')
        + b"\x48\x31\xf6"
        + b"\x48\xc7\xc2" + buffer_size.to_bytes(4, 'little')
        + b"\x0f\x05"
    )

# Reconnaissance Genes

def gene_scan_local_ports(start_port=20, end_port=1024, timeout=0.2):
    """Scan ports on localhost to find open services."""
    open_ports = []
    for port in range(start_port, end_port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        result = s.connect_ex(('127.0.0.1', port))
        if result == 0:
            open_ports.append(port)
        s.close()
    return open_ports

def gene_scan_network_interfaces():
    """Enumerate available network interfaces and IPs."""
    interfaces = psutil.net_if_addrs()
    results = {}
    for iface, addrs in interfaces.items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                results[iface] = addr.address
    return results

def gene_check_socket_response(port, timeout=0.5):
    """Attempt to connect to a specific port to check for response."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(('127.0.0.1', port))
        s.close()
        return True
    except Exception:
        s.close()
        return False

# Gene dispatchers
NETWORKING_GENE_FUNCTIONS = [
    gene_open_socket,
    gene_connect,
    gene_send_shellcode,
    gene_recv_shellcode,
]

RECON_GENE_FUNCTIONS = [
    gene_scan_local_ports,
    gene_scan_network_interfaces,
    gene_check_socket_response,
]

def get_random_networking_gene():
    func = random.choice(NETWORKING_GENE_FUNCTIONS)
    return func()

def list_networking_genes():
    return [func.__name__ for func in NETWORKING_GENE_FUNCTIONS]

def list_recon_genes():
    return [func.__name__ for func in RECON_GENE_FUNCTIONS]

# Example usage
if __name__ == "__main__":
    print("Available networking genes:", list_networking_genes())
    print("Available reconnaissance genes:", list_recon_genes())
    fragment = get_random_networking_gene()
    print(f"Generated networking gene fragment: {fragment.hex()}")
