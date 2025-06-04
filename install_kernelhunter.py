#!/usr/bin/env python3
"""Installer script for KernelHunter.

This Python installer replicates the functionality of the previous
``install_kernelhunter.sh`` script. It sets up configuration directories,
creates an initial configuration file, copies the main script to the
user's data directory and creates a launcher in ``~/.local/bin``.
"""

import argparse
import json
import os
import shutil
import sys


def ask_scope() -> str:
    """Interactively ask the user for the configuration scope."""
    print()
    print("Choose configuration scope:")
    print(" 1) Global (/etc/kernelhunter)")
    print(" 2) Local  (~/.config/kernelhunter)")
    choice = input("Scope [1/2]: ").strip()
    return "global" if choice == "1" else "local"


def ensure_dirs(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install KernelHunter")
    parser.add_argument(
        "--scope",
        choices=["global", "local"],
        help="Configuration scope (global or local)",
    )
    args = parser.parse_args()

    scope = args.scope or ask_scope()

    home = os.path.expanduser("~")
    if scope == "global":
        config_dir = "/etc/kernelhunter"
        reservoir_dir = "/var/lib/kernelhunter/reservoir"
    else:
        config_dir = os.path.join(home, ".config", "kernelhunter")
        reservoir_dir = os.path.join(home, ".local", "share", "kernelhunter", "reservoir")

    ensure_dirs(config_dir)
    ensure_dirs(reservoir_dir)

    try:
        if scope == "global":
            os.chmod(reservoir_dir, 0o1777)
        else:
            os.chmod(reservoir_dir, 0o700)
    except PermissionError:
        print(f"[!] Could not set permissions on {reservoir_dir}.")

    config = {"reservoir_path": reservoir_dir, "openai_api_key": ""}
    config_file = os.path.join(config_dir, "config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    try:
        os.chmod(config_file, 0o600)
    except PermissionError:
        print(f"[!] Could not set permissions on {config_file}.")

    install_dir = os.path.join(home, ".local", "share", "kernelhunter")
    bin_dir = os.path.join(home, ".local", "bin")
    ensure_dirs(install_dir)
    ensure_dirs(bin_dir)

    script_src = "kernelhunter.py"
    if not os.path.exists(script_src):
        # Fallback to camel case name used in the repository
        script_src = "kernelHunter.py"
    if not os.path.exists(script_src):
        print("[-] kernelhunter.py not found.")
        sys.exit(1)

    script_dst = os.path.join(install_dir, "kernelhunter.py")
    shutil.copy(script_src, script_dst)
    os.chmod(script_dst, 0o755)

    symlink_path = os.path.join(bin_dir, "kernelhunter")
    try:
        if os.path.islink(symlink_path) or os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(script_dst, symlink_path)
    except OSError as exc:
        print(f"[!] Could not create symlink {symlink_path}: {exc}")

    path_env = os.environ.get("PATH", "")
    if bin_dir not in path_env.split(os.pathsep):
        bashrc = os.path.join(home, ".bashrc")
        with open(bashrc, "a", encoding="utf-8") as f:
            f.write(f"\nexport PATH=\"$PATH:{bin_dir}\"\n")
        print(f"[*] Added {bin_dir} to PATH in {bashrc}")

    ensure_dirs(os.path.join(install_dir, "logs"))

    print("\n[+] Installation complete.")
    print("    You can now run the tool with:")
    print("    kernelhunter")
    print("\n    NOTE: If 'kernelhunter' is not found, try opening a new terminal or run:")
    print("    source ~/.bashrc")


if __name__ == "__main__":
    main()
