#!/usr/bin/env python3
"""KernelHunter configuration management module.

This module centralizes configuration settings for KernelHunter in
``/etc/kernelhunter/config.json``. The configuration stores the path
of the genetic reservoir directory and the OpenAI API key.
"""

import argparse
import json
import os
from typing import Any, Dict

CONFIG_DIR = "/etc/kernelhunter"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

DEFAULT_CONFIG = {
    "reservoir_path": "/var/lib/kernelhunter/reservoir",
    "openai_api_key": ""
}


def load_config() -> Dict[str, Any]:
    """Load configuration from the config file.

    Returns the configuration with defaults applied if the file does not exist.
    """
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG.copy()

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        # Fallback to defaults on any read/parse error
        return DEFAULT_CONFIG.copy()

    cfg = DEFAULT_CONFIG.copy()
    cfg.update({k: v for k, v in data.items() if k in DEFAULT_CONFIG})
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    """Save configuration to the config file."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except PermissionError:
        # Ignore permission errors when running without privileges
        pass


def get_reservoir_dir() -> str:
    """Return the configured reservoir directory."""
    return load_config()["reservoir_path"]


def get_reservoir_file(filename: str = "kernelhunter_reservoir.pkl") -> str:
    """Return the full path to a reservoir file."""
    return os.path.join(get_reservoir_dir(), filename)


def get_api_key() -> str:
    """Return the configured OpenAI API key."""
    cfg = load_config()
    return cfg.get("openai_api_key", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="KernelHunter configuration")
    parser.add_argument("--show", action="store_true", help="Display current configuration")
    parser.add_argument("--reservoir-path", help="Set reservoir directory path")
    parser.add_argument("--api-key", help="Set OpenAI API key")
    args = parser.parse_args()

    config = load_config()

    if args.reservoir_path:
        config["reservoir_path"] = args.reservoir_path

    if args.api_key is not None:
        config["openai_api_key"] = args.api_key

    if args.reservoir_path or args.api_key is not None:
        save_config(config)

    if args.show or not (args.reservoir_path or args.api_key is not None):
        print(json.dumps(config, indent=4))


if __name__ == "__main__":
    main()
