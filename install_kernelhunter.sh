#!/bin/bash

set -e

echo "[*] Installing KernelHunter..."

# === WARNING ===
echo "=============================================================="
echo "[!] WARNING: It is strongly recommended to NEVER run KernelHunter as root."
echo "    Running fuzzed shellcode as root may compromise your system."
echo "=============================================================="
sleep 2

# Check if user is root
if [[ "$EUID" -eq 0 ]]; then
    echo "[!] Detected root user."
    echo "    It's highly discouraged to install or run this tool as root."
    read -p "    Do you want to continue anyway? (y/N): " choice
    if [[ "$choice" != "y" && "$choice" != "Y" ]]; then
        echo "[-] Aborting installation."
        exit 1
    fi
fi

# Installation target
INSTALL_DIR="$HOME/.local/share/kernelhunter"
BIN_DIR="$HOME/.local/bin"
SCRIPT_NAME="kernelhunter.py"
EXECUTABLE_NAME="kernelhunter"
SYMLINK_PATH="$BIN_DIR/$EXECUTABLE_NAME"

# Create directories
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"

# Copy the main script
cp "$SCRIPT_NAME" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/$SCRIPT_NAME"

# Create the symlink to local bin
ln -sf "$INSTALL_DIR/$SCRIPT_NAME" "$SYMLINK_PATH"

# Optionally add local bin to PATH if missing
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "[*] Adding $BIN_DIR to PATH..."
    echo "export PATH=\"\$PATH:$BIN_DIR\"" >> "$HOME/.bashrc"
    export PATH="$PATH:$BIN_DIR"
fi

# Optional local log directory
mkdir -p "$HOME/.local/share/kernelhunter/logs"

echo ""
echo "[+] Installation complete."
echo "    You can now run the tool with:"
echo "    $EXECUTABLE_NAME"
echo ""
echo "    NOTE: If 'kernelhunter' is not found, try opening a new terminal or run:"
echo "    source ~/.bashrc"
