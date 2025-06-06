# KernelHunter

**KernelHunter** is an evolutionary fuzzer designed to discover crashes and vulnerabilities in operating systems by generating and executing x86_64 shellcodes. Unlike traditional fuzzers, KernelHunter uses a genetic algorithm approach to mutate and select the most effective shellcodes over multiple generations.

---

## What KernelHunter Does

KernelHunter operates by generating random shellcodes, compiling and executing them in a controlled environment, and analyzing the results for crashes and anomalies. It captures low-level execution failures such as segmentation faults (SIGSEGV), illegal instructions (SIGILL), floating-point exceptions (SIGFPE), and traps (SIGTRAP), classifying and prioritizing them based on severity and impact.

What sets KernelHunter apart is its evolutionary engine. Rather than relying on blind mutation, it evolves shellcodes across generations, favoring those that cause the most interesting system-level behaviors. This makes it especially effective for identifying edge-case vulnerabilities that are often missed by traditional fuzzers.

---

## Key Features

- Generation and mutation of x86_64 shellcodes
- Safe execution of binaries with timeout control
- Crash capture and classification: SIGSEGV, SIGILL, SIGFPE, SIGTRAP
- Detection of potential system-level impacts (e.g., critical signal patterns)
- Per-generation statistical analysis:
  - Crash rate
  - Average shellcode length
  - Most frequent byte patterns
- Structured JSON output for crash metadata
- Post-run analysis of critical crash segments
- Demo mode for reproducible runs and showcases

---

## Why It's Useful

KernelHunter provides a practical, autonomous approach to exploring how low-level input (shellcode) can destabilize an operating system or uncover flaws in the kernel’s memory handling. It’s ideal for:

- Security researchers looking for novel syscall-level bugs
- Kernel developers stress-testing system behavior
- Offensive security professionals creating realistic crash-based PoCs
- Dataset generation for ML-based exploit detection models

Because it doesn't rely on any hardcoded corpus or known exploit templates, KernelHunter can produce previously unseen shellcode patterns that lead to real impact. This positions it as a creative tool in vulnerability discovery pipelines.

---


## Installation

KernelHunter is a standalone Python script and can be installed without root access.

> ⚠️ **WARNING:** It is strongly recommended to **never run KernelHunter as root**. Fuzzing shellcode with elevated privileges can lead to complete system compromise.

### Step-by-step Installation

```bash
# 1. Clone the repository
git clone https://github.com/openbashok/kernelhunter
cd kernelhunter

# 2. Run the installer (no root required)
python install_kernelhunter.py
#    (you will be asked for your OpenAI API key)

# 3. (Optional) Restart your terminal or run:
source ~/.bashrc

# 4. Run it from anywhere:
kernelhunter
```
The previous `install_kernelhunter.sh` script is now deprecated.

After installation, KernelHunter will be available under `~/.local/bin/kernelhunter` for the current user. During installation you can choose whether the configuration is stored globally in `/etc/kernelhunter` or locally in `~/.config/kernelhunter`.
The installer will also create an empty genetic reservoir if none is present.

If you choose a **global** installation, this reservoir lives in
`/var/lib/kernelhunter/reservoir` and the directory is created with mode
`1777`, allowing all users to write to it (similar to `/tmp`). This makes the
reservoir shared across accounts but may not be desirable in hardened
environments. A **local** installation (`--scope local`) keeps the reservoir
private under your home directory.

## Configuration

KernelHunter keeps its configuration in either `/etc/kernelhunter/config.json` or
`~/.config/kernelhunter/config.json` depending on your installation choice. The
file is created automatically during installation and can be inspected with:

```bash
$ python kernelhunter_config.py --show
```

The configuration contains the path to the genetic reservoir directory and the
OpenAI API key used by the analysis modules. You can update these values with:

```bash
# Use sudo only if modifying the global configuration
python kernelhunter_config.py --reservoir-path /custom/reservoir
python kernelhunter_config.py --api-key YOUR_OPENAI_KEY
```

The `OPENAI_API_KEY` environment variable still takes precedence over the value
stored in the configuration file.

To enable the experimental reinforcement learning mode that adapts attack and
mutation probabilities over time, set the `use_rl_weights` flag:

```bash
python kernelhunter_config.py --use-rl-weights
# Optionally provide custom starting weights
python kernelhunter_config.py --attack-weights 2,1,1,6,5,...
python kernelhunter_config.py --mutation-weights 22,12,18,18,10,8,8,4
```

When enabled, updated weights are recorded in `kernelhunter_metrics.json` after
each generation.



## Quick Start

```bash
$ python kernelhunter.py
```




## Sample Output

```text
kernelhunter@localhost:~/demo2$ kernelhunter

  _  __                    _   _   _             _
 | |/ /___ _ __ _ __   ___| | | | | |_   _ _ __ | |_ ___ _ __
 | ' // _ \ '__| '_ \ / _ \ | | |_| | | | | '_ \| __/ _ \ '__|
 | . \  __/ |  | | | |  __/ | |  _  | |_| | | | | ||  __/ |
 |_|\_\___|_|  |_| |_|\___|_| |_| |_|\__,_|_| |_|\__\___|_|

 Fuzzer evolutivo para vulnerabilidades del sistema operativo
 -------------------------------------------------------------

Configuración: 20 programas por generación, timeout: 3s

Generación 0/10000 (población: 1)
Crash 0000: SIGSEGV | Shellcode: 3348c7c03c0000004831ff0f05
Crash 0001: SIGSEGV | Shellcode: 488bd377
Crash 0002: SIGSEGV | Shellcode: 4831d284674548c7c03c0000004831ff0f05
Crash 0004: SIGSEGV | Shellcode: 4831c9de
Crash 0006: SIGSEGV | Shellcode: 5aed48c7c03c0000004831ff0f05
[SYSTEM IMPACT] Crash 0007: SIGILL | Shellcode: e1f6475136d648c7c03c0000004831ff0f05
Crash 0008: SIGSEGV | Shellcode: 709a0848c7c03c0000004831ff0f05
Crash 0010: SIGSEGV | Shellcode: 48c7c080
Crash 0011: SIGSEGV | Shellcode: b8eb1f48c7c03c0000004831ff0f05
Crash 0012: SIGSEGV | Shellcode: 4889b448c7c03c0000004831ff0f05
[SYSTEM IMPACT] Crash 0013: SIGILL | Shellcode: 4831ff5716577c
Crash 0014: SIGSEGV | Shellcode: 4831d2e2f5
Crash 0015: SIGSEGV | Shellcode: a0ff48c7c03c0000004831ff0f05
Crash 0017: SIGSEGV | Shellcode: 4d31c9a30348c7c03c0000004831ff0f05
Crash 0018: SIGSEGV | Shellcode: ff36
[GEN 0] Crash rate: 75.0% | Sys impacts: 2 | Avg length: 12.2
[GEN 0] Crash types: {'SIGNAL_SIGSEGV': 13, 'SIGNAL_SIGILL': 2}

Generación 1/10000 (población: 5)
Crash 0000: SIGSEGV | Shellcode: 5d0bc520a8cb1d3f0a48c7c03c0000004831ff0f05
[SYSTEM IMPACT] Crash 0001: SIGILL | Shellcode: 37a4630f0548c7c03c0000004831ff0f05
Crash 0004: SIGSEGV | Shellcode: b80bc548c7c03c0000004831ff0f05
Crash 0005: SIGSEGV | Shellcode: 5d0bb6ffff7c28f2c548c7c03c0000004831ff0f05
Crash 0007: SIGSEGV | Shellcode: 488b02515d0bc548c7c03c0000004831ff0f05
[SYSTEM IMPACT] Crash 0008: SIGILL | Shellcode: 56ba4831d2a416b8ee8c20e848c7c03c0000004831ff0f05
[SYSTEM IMPACT] Crash 0010: SIGILL | Shellcode: 2f2e540d626e0f0548c7c03c0000004831ff0f05
Crash 0011: SIGSEGV | Shellcode: 484d31c9cfd0cf31d20cddf248c7c03c0000004831ff0f05
Crash 0012: SIGSEGV | Shellcode: 0f05cd80
[SYSTEM IMPACT] Crash 0013: SIGILL | Shellcode: 60c37face50f0548c7c03c0000004831ff0f05
[SYSTEM IMPACT] Crash 0014: SIGILL | Shellcode: 0fb94f05
Crash 0015: SIGSEGV | Shellcode: 4831ffdd0f0548c7c03c0000004831ff0f05
Crash 0016: SIGSEGV | Shellcode: e4750e2b8b4831d20cddf248c7c03c0000004831ff0f05
Crash 0017: SIGSEGV | Shellcode: 56ba4831ff00542e8c20e8
Crash 0018: SIGSEGV | Shellcode: 56ba8c204d31c99b2258e848c7c03c0000004831ff0f05
Crash 0019: SIGSEGV | Shellcode: 765d0bc548c7c03c0000004831ff0f05
[GEN 1] Crash rate: 80.0% | Sys impacts: 5 | Avg length: 17.8
[GEN 1] Crash types: {'SIGNAL_SIGSEGV': 11, 'SIGNAL_SIGILL': 5}
```





## Menu interface

KernelHunter includes a curses-based menu for launching the different tools. Start it with:

```bash
python menu.py
```

Use the arrow keys and Enter to select one of the following options:

- Run KernelHunter Fuzzer
- Crash Explorer
- Reservoir Manager
- KernelHunter Monitor
- Attack/Mutation Stats
- Kernel Error Dashboard
- Quit

When a tool exits you will be returned to the menu.

## OllyDbg-style interface

For a split-screen view inspired by OllyDbg, use the alternative menu:

```bash
python ollydbg_menu.py
```

The left panel lists available tools and the right panel shows their output.
Use the arrow keys to navigate, Enter to run the selected tool, and `q` to quit.

## tmux launcher

The repository includes `tmux_launcher.sh` for quickly opening the monitoring tools in a grid of tmux panes. It requires `tmux` to be installed on your system.

Run it from the repository root with:

```bash
./tmux_launcher.sh
```

This will start the KernelHunter monitor, the reservoir UI, the crash UI, and the dashboard in one tmux session arranged in two columns.


