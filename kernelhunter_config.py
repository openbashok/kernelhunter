#!/usr/bin/env python3
"""KernelHunter configuration management module.

This module handles configuration for KernelHunter. The settings can be
stored either globally in ``/etc/kernelhunter/config.json`` or locally in
``~/.config/kernelhunter/config.json``. The configuration contains the path
of the genetic reservoir directory and the OpenAI API key.
"""

import argparse
import json
import os
from typing import Any, Dict

SYSTEM_CONFIG_DIR = "/etc/kernelhunter"
USER_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "kernelhunter")
USER_RESERVOIR_DIR = os.path.join(os.path.expanduser("~"), ".local", "share", "kernelhunter", "reservoir")

DEFAULT_SYSTEM_CONFIG = {
    "reservoir_path": "/var/lib/kernelhunter/reservoir",
    "openai_api_key": "",
    "use_rl_weights": False,
    "attack_weights": None,
    "mutation_weights": None
}

DEFAULT_USER_CONFIG = {
    "reservoir_path": USER_RESERVOIR_DIR,
    "openai_api_key": "",
    "use_rl_weights": False,
    "attack_weights": None,
    "mutation_weights": None
}


def get_config_dir() -> str:
    """Return the directory where the configuration file is stored."""
    override = os.getenv("KERNELHUNTER_CONFIG_DIR")
    if override:
        return override

    system_cfg = os.path.join(SYSTEM_CONFIG_DIR, "config.json")
    if os.path.exists(system_cfg):
        return SYSTEM_CONFIG_DIR
    return USER_CONFIG_DIR


def get_default_config() -> Dict[str, Any]:
    """Return default configuration for the detected scope."""
    if get_config_dir() == SYSTEM_CONFIG_DIR:
        return DEFAULT_SYSTEM_CONFIG
    return DEFAULT_USER_CONFIG


def get_config_file() -> str:
    """Return full path to the configuration file."""
    return os.path.join(get_config_dir(), "config.json")


def load_config() -> Dict[str, Any]:
    """Load configuration from the config file with sane defaults."""
    filename = get_config_file()
    default_cfg = get_default_config()

    if not os.path.exists(filename):
        return default_cfg.copy()

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return default_cfg.copy()

    cfg = default_cfg.copy()
    cfg.update({k: v for k, v in data.items() if k in default_cfg})
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    """Save configuration to the config file."""
    filename = get_config_file()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)
    try:
        os.chmod(filename, 0o600)
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
    parser.add_argument("--use-rl-weights", action="store_true", help="Enable reinforcement learning weights")
    parser.add_argument("--attack-weights", help="Comma separated list of attack weights")
    parser.add_argument("--mutation-weights", help="Comma separated list of mutation weights")
    args = parser.parse_args()

    config = load_config()

    if args.reservoir_path:
        config["reservoir_path"] = args.reservoir_path

    if args.api_key is not None:
        config["openai_api_key"] = args.api_key

    if args.use_rl_weights:
        config["use_rl_weights"] = True

    if args.attack_weights:
        config["attack_weights"] = [int(x) for x in args.attack_weights.split(',') if x]

    if args.mutation_weights:
        config["mutation_weights"] = [int(x) for x in args.mutation_weights.split(',') if x]

    if args.reservoir_path or args.api_key is not None or args.use_rl_weights or args.attack_weights or args.mutation_weights:
        save_config(config)

    if args.show or not (args.reservoir_path or args.api_key is not None or args.use_rl_weights or args.attack_weights or args.mutation_weights):
        print(json.dumps(config, indent=4))


if __name__ == "__main__":
    main()
