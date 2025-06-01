"""
Module: tcp_shellcode_collector.py

Description:
Shellcode Sampler for KernelHunter.
Listens on a TCP socket, receives shellcode payloads from remote cells,
saves them automatically into the `received_shellcodes/` folder for future evolutionary cycles.
This module acts as an external genetic memory system, collecting
samples independently from the executing shellcodes.

Probable Impact:
- High - Expands the evolutionary gene pool with real-world propagated shellcodes.
- Enables tracking of evolving payloads across generations.

Risk Level:
⚡ Moderate - Minimal execution risk, acts only as passive collector.
"""

import socket
import os
import time

RECEIVED_DIR = "received_shellcodes"
LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 9001
BUFFER_SIZE = 4096


def ensure_received_dir():
    """Ensure the output directory exists."""
    os.makedirs(RECEIVED_DIR, exist_ok=True)


def save_shellcode(payload):
    """Save the received shellcode payload to a file."""
    timestamp = int(time.time())
    filename = os.path.join(RECEIVED_DIR, f"shellcode_{timestamp}.bin")
    with open(filename, "wb") as f:
        f.write(payload)
    print(f"[+] Shellcode saved to {filename}")


def start_sampler():
    """Start the Shellcode Sampler server."""
    ensure_received_dir()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((LISTEN_IP, LISTEN_PORT))
        server.listen(5)
        print(f"[*] Listening on {LISTEN_IP}:{LISTEN_PORT}")

        while True:
            client, addr = server.accept()
            print(f"[+] Connection from {addr}")
            with client:
                payload = client.recv(BUFFER_SIZE)
                if payload:
                    save_shellcode(payload)
                else:
                    print("[-] Empty payload received.")


# Example usage
if __name__ == "__main__":
    start_sampler()
