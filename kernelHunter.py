#!/usr/bin/env python3
# KernelHunter - Fuzzer evolutivo para encontrar vulnerabilidades en el sistema operativo
# Mariano Somoza
# Un enfoque evolutivo para generar shellcodes que potencialmente puedan impactar
# la estabilidad del sistema operativo.


import os
import subprocess
import random
import time
import signal
import sys
import json
import asyncio
import concurrent.futures
import multiprocessing
from random import randint, choice
from collections import Counter, defaultdict, deque
from genetic_reservoir import GeneticReservoir
from performance_logger import PerformanceLogger
try:
    from kernelhunter_config import get_reservoir_file, load_config
except Exception:
    def get_reservoir_file(name="kernelhunter_reservoir.pkl"):
        return name
    def load_config():
        return {}
from memory_pressure_mutation import generate_memory_pressure_fragment
from advanced_crossover import crossover_shellcodes_advanced
from cache_pollution_attack import generate_cache_pollution_fragment
from control_flow_traps import generate_control_flow_trap_fragment
from crispr_mutation import crispr_edit_shellcode
from deep_rop_chain_generator import generate_deep_rop_chain_fragment
from dma_confusion import generate_dma_confusion_fragment
from duplication_mutation import duplicate_fragment
from entropy_drain_attack import generate_entropy_drain_fragment
from external_adn import get_random_fragment_from_bin
from extract_function_adn import get_function_or_fragment
from filesystem_chaos import generate_filesystem_chaos_fragment
from gene_bank import get_random_gene
from gene_bank_advanced_v2 import get_random_gene_dynamic
from hyper_advanced_memory_corruptor import generate_hyper_advanced_corruptor_fragment
from interrupt_storm import generate_interrupt_storm_fragment
from inversion_mutation import invert_fragment
from ipc_stress_attack import generate_ipc_stress_fragment
from kpti_breaker import generate_kpti_breaker_fragment
from memory_fragmentation_attack import generate_memory_fragmentation_fragment
from module_loading_storm import generate_module_loading_storm_fragment
from network_stack_fuzz import generate_network_stack_fuzz_fragment
from neutral_mutation import insert_neutral_mutation
from nop_islands import generate_nop_island, reset_nop_counter
from page_fault_flood import generate_page_fault_flood_fragment
from pointer_attack_mutation import generate_pointer_attack_fragment
from privileged_cpu_destruction import generate_privileged_cpu_destruction_fragment
from privileged_storm import generate_privileged_storm_fragment
from resource_starvation_attack import generate_resource_starvation_fragment
from scheduler_attack import generate_scheduler_attack_fragment
from shadow_memory_corruptor import generate_shadow_memory_corruptor_fragment
from smap_smep_bypass import generate_smap_smep_bypass_fragment
from speculative_confusion import generate_speculative_confusion_fragment
from syscall_reentrancy_storm import generate_syscall_reentrancy_storm_fragment
from syscall_storm import generate_syscall_storm
from syscall_table_stress import generate_syscall_table_stress_fragment
from transposition_mutation_nop import transpose_fragment_nop_aware
from ultimate_kernel_panic_generator import generate_ultimate_panic_fragment
from ai_shellcode_mutation import generate_ai_shellcode_fragment

# Performance logging
pythonlogger = None

# Simple ANSI color helpers for terminal output
ANSI_COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "reset": "\033[0m",
}


def color_text(text, color):
    """Return text wrapped in ANSI color codes."""
    return f"{ANSI_COLORS.get(color, '')}{text}{ANSI_COLORS['reset'] if color else ''}"

def format_shellcode_c_array(shellcode_bytes):
    return ','.join(f'0x{b:02x}' for b in shellcode_bytes)

# Instrucciones base más interesantes para interactuar con el SO
BASE_SHELLCODE = b""
EXIT_SYSCALL = b"\x48\xc7\xc0\x3c\x00\x00\x00\x48\x31\xff\x0f\x05"  # syscall exit

# Syscalls comunes que podrían ser interesantes para probar
SYSCALL_PATTERN = b"\x0f\x05"  # instrucción syscall x86_64
SYSCALL_SETUP = [
    b"\x48\xc7\xc0", # mov rax, X (syscall number)
    b"\x48\x31\xff", # xor rdi, rdi
    b"\x48\x31\xf6", # xor rsi, rsi
    b"\x48\x31\xd2", # xor rdx, rdx
    b"\x48\x31\xc9", # xor rcx, rcx
    b"\x49\x89\xca", # mov r10, rcx
    b"\x4d\x31\xc9", # xor r9, r9
    b"\x4d\x31\xc0", # xor r8, r8
]

# Configuración
OUTPUT_DIR = "kernelhunter_generations"
NUM_PROGRAMS = 50
MAX_GENERATIONS = 1000
TIMEOUT = 3
LOG_FILE = "kernelhunter_survivors.txt"
CRASH_LOG = "kernelhunter_crashes.txt"
METRICS_FILE = "kernelhunter_metrics.json"
CHECKPOINT_INTERVAL = 10
MAX_POPULATION_SIZE = 1000  # Límite por recursos
NATURAL_SELECTION_MODE = True  # Selección natural sin sesgos

# Crear directorios necesarios
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("kernelhunter_crashes", exist_ok=True)
os.makedirs("kernelhunter_critical", exist_ok=True)

# Load configuration for reinforcement learning
cfg = load_config()
USE_RL_WEIGHTS = cfg.get("use_rl_weights", False)

# Attack options and weights
ATTACK_OPTIONS = [
    "random_bytes",
    "syscall_setup",
    "syscall",
    "memory_access",
    "privileged",
    "arithmetic",
    "control_flow",
    "x86_opcode",
    "simd",
    "known_vulns",
    "segment_registers",
    "speculative_exec",
    "forced_exception",
    "control_registers",
    "stack_manipulation",
    "full_kernel_syscall",
    "memory_pressure",
    "cache_pollution",
    "control_flow_trap",
    "deep_rop_chain",
    "dma_confusion",
    "entropy_drain",
    "external_adn",
    "function_adn",
    "filesystem_chaos",
    "gene_bank",
    "gene_bank_dynamic",
    "hyper_corruptor",
    "interrupt_storm",
    "ipc_stress",
    "kpti_breaker",
    "memory_fragmentation",
    "module_loading_storm",
    "network_stack_fuzz",
    "neutral_mutation",
    "nop_island",
    "page_fault_flood",
    "pointer_attack",
    "privileged_cpu_destruction",
    "privileged_storm",
    "resource_starvation",
    "scheduler_attack",
    "shadow_corruptor",
    "smap_smep_bypass",
    "speculative_confusion",
    "syscall_reentrancy_storm",
    "syscall_storm",
    "syscall_table_stress",
    "ultimate_panic",
    "ai_shellcode",
]

DEFAULT_ATTACK_WEIGHTS = [
    80, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
]

MUTATION_TYPES = [
    "add",
    "remove",
    "modify",
    "duplicate",
    "mass_duplicate",
    "invert",
    "transpose_nop",
    "crispr",
]

DEFAULT_MUTATION_WEIGHTS = [90, 3, 2, 1, 1, 1, 1, 1]

# Totals used to keep weights balanced when reinforcement learning is enabled
DEFAULT_ATTACK_WEIGHT_SUM = sum(DEFAULT_ATTACK_WEIGHTS)
DEFAULT_MUTATION_WEIGHT_SUM = sum(DEFAULT_MUTATION_WEIGHTS)

# Maximum possible reward used for normalization
MAX_REWARD = 15

# Exploration factor for epsilon-greedy selection
def get_epsilon(generation):
    if generation < 20:
        return 0.3
    else:
        return max(0.05, 0.3 * (0.95 ** (generation - 20)))

attack_weights = cfg.get("attack_weights") or DEFAULT_ATTACK_WEIGHTS.copy()
if len(attack_weights) != len(ATTACK_OPTIONS):
    attack_weights = DEFAULT_ATTACK_WEIGHTS.copy()

mutation_weights = cfg.get("mutation_weights") or DEFAULT_MUTATION_WEIGHTS.copy()
if len(mutation_weights) != len(MUTATION_TYPES):
    mutation_weights = DEFAULT_MUTATION_WEIGHTS.copy()

attack_index = {name: i for i, name in enumerate(ATTACK_OPTIONS)}
mutation_index = {name: i for i, name in enumerate(MUTATION_TYPES)}

# Q-values and counters for epsilon-greedy bandit
def initialize_bandit_values():
    """Random optimistic initialization to prevent premature convergence."""
    import random
    attack_values = [random.uniform(5.0, 8.0) for _ in ATTACK_OPTIONS]
    mutation_values = [random.uniform(4.0, 7.0) for _ in MUTATION_TYPES]
    return attack_values, mutation_values

attack_q_values, mutation_q_values = initialize_bandit_values()
attack_counts = [0] * len(ATTACK_OPTIONS)
mutation_counts = [0] * len(MUTATION_TYPES)

last_attack_type = None
last_mutation_type = None
attack_success = Counter()
mutation_success = Counter()
current_generation = 0

# Sliding window trackers for reinforcement learning rewards
class SlidingWindowReward:
    def __init__(self, window_size=100):
        self.success_history = defaultdict(deque)
        self.window_size = window_size

    def add_result(self, action, reward):
        dq = self.success_history[action]
        dq.append(reward)
        if len(dq) > self.window_size:
            dq.popleft()

    def success_rate(self, action):
        dq = self.success_history.get(action)
        if not dq:
            return 0.0
        return (sum(dq) / len(dq)) / MAX_REWARD

    def overall_rate(self):
        total_reward = sum(sum(dq) for dq in self.success_history.values())
        total_count = sum(len(dq) for dq in self.success_history.values())
        if total_count == 0:
            return 0.0
        return (total_reward / total_count) / MAX_REWARD


sliding_attack_reward = SlidingWindowReward(window_size=50)
sliding_mutation_reward = SlidingWindowReward(window_size=50)



syscalls = {}
with open("/usr/include/x86_64-linux-gnu/asm/unistd_64.h", "r") as f:
    for line in f:
        if line.startswith("#define __NR_"):
            parts = line.split()
            syscall_name = parts[1][5:]  # elimina "__NR_"
            syscall_num = int(parts[2])
            syscalls[syscall_num] = syscall_name


#Inicializo el reservorio genetico
# Initialize the genetic reservoir at the beginning of your main function or at module level
from genetic_reservoir import GeneticReservoir
genetic_reservoir = GeneticReservoir(max_size=200, diversity_threshold=0.3)

# Optionally try to load a previous reservoir
try:
    genetic_reservoir.load_from_file(get_reservoir_file())
    print(f"Loaded genetic reservoir with {len(genetic_reservoir)} individuals")
except:
    print("Starting with a new genetic reservoir")


def interpret_instruction(instruction_bytes):
    """Intenta interpretar instrucciones comunes de x86_64"""
    if len(instruction_bytes) >= 12 and instruction_bytes.endswith(b"\x48\x31\xff\x0f\x05"):
        return "syscall exit"

    # Identificar syscalls comunes
    if len(instruction_bytes) >= 4 and instruction_bytes.startswith(b"\x48\xc7\xc0"):
        syscall_num = instruction_bytes[3]
        syscall_names = {
            0: "read",
            1: "write",
            2: "open",
            3: "close",
            9: "mmap",
            10: "mprotect",
            11: "munmap",
            60: "exit",
            105: "setuid",
            231: "exit_group"
        }
        if syscall_num in syscall_names:
            return f"syscall setup for {syscall_names[syscall_num]}"
        else:
            return f"syscall setup for syscall #{syscall_num}"

    # Identificar syscall
    if instruction_bytes == b"\x0f\x05":
        return "syscall instruction"

    # Identificar otras instrucciones comunes
    instruction_map = {
        b"\x48\x31\xff": "xor rdi, rdi",
        b"\x48\x31\xf6": "xor rsi, rsi",
        b"\x48\x31\xd2": "xor rdx, rdx",
        b"\x48\x31\xc9": "xor rcx, rcx",
        b"\x49\x89\xca": "mov r10, rcx",
        b"\x4d\x31\xc9": "xor r9, r9",
        b"\x4d\x31\xc0": "xor r8, r8",
        b"\xf4": "hlt",
        b"\xcd\x80": "int 0x80 (32-bit syscall)",
        b"\x0f\x01\xf8": "swapgs",
        b"\x0f\x01\xd0": "xgetbv",
        b"\x0f\x01\xd4": "vmenter",
        b"\x0f\x01\xd5": "vmexit",
        b"\x0f\x30": "wrmsr",
        b"\x0f\x32": "rdmsr",
        b"\x0f\x34": "sysenter",
        b"\x0f\x35": "sysexit",
        b"\xc3": "ret",
    }

    for pattern, desc in instruction_map.items():
        if instruction_bytes.startswith(pattern):
            return desc

    return "instrucción desconocida"

def print_shellcode_hex(shellcode, max_bytes=32, escape_format=True):
    """
    Imprime un shellcode en formato hexadecimal de forma legible
    Si escape_format es True, usa el formato \x01\x02
    Si es False, usa el formato 01 02
    """
    if escape_format:
        if len(shellcode) <= max_bytes:
            return ''.join(f'\\x{b:02x}' for b in shellcode)
        else:
            # Imprimir los primeros max_bytes/2 bytes
            start = ''.join(f'\\x{b:02x}' for b in shellcode[:max_bytes//2])

            # Imprimir los últimos max_bytes/2 bytes
            end = ''.join(f'\\x{b:02x}' for b in shellcode[-(max_bytes//2):])

            return f"{start}... (omitted {len(shellcode) - max_bytes} bytes) ...{end}"
    else:
        if len(shellcode) <= max_bytes:
            hex_str = shellcode.hex()
            formatted = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
            return formatted
        else:
            # Imprimir los primeros max_bytes/2 bytes
            hex_start = shellcode[:max_bytes//2].hex()
            formatted_start = ' '.join(hex_start[i:i+2] for i in range(0, len(hex_start), 2))

            # Imprimir los últimos max_bytes/2 bytes
            hex_end = shellcode[-(max_bytes//2):].hex()
            formatted_end = ' '.join(hex_end[i:i+2] for i in range(0, len(hex_end), 2))

            return f"{formatted_start} ... (omitted {len(shellcode) - max_bytes} bytes) ... {formatted_end}"

# Reinforcement learning helpers
def select_with_epsilon_greedy(options, q_values, epsilon=0.1):
    """Select an option using epsilon-greedy strategy with deterministic exploitation."""
    if random.random() < epsilon:
        return random.choice(options)
    best_idx = max(range(len(q_values)), key=lambda i: q_values[i])
    return options[best_idx]

def update_q_value(q_values, counts, index, reward):
    """Incrementally update estimated Q-value for a given action."""
    counts[index] += 1
    q_values[index] += (reward - q_values[index]) / counts[index]


def calculate_reward(crash_info):
    reward = 0
    if crash_info.get("system_impact"):
        reward += 10
    ctype = crash_info.get("crash_type", "")
    if "SIGILL" in ctype:
        reward += 5
    elif "SIGSEGV" in ctype:
        reward += 3
    if "TIMEOUT" in ctype:
        reward -= 2
    return max(0, reward)

# Métricas para seguimiento
metrics = {
    "generations": [],
    "crash_rates": [],
    "crash_types": {},
    "system_impacts": [],
    "shellcode_lengths": [],
    # Totals across the entire fuzzing run
    "attack_totals": {},
    "mutation_totals": {},
    "attack_weights_history": [],
    "mutation_weights_history": [],
}

# Counters for attack and mutation statistics per generation
attack_counter = Counter()
mutation_counter = Counter()
generation_attack_counter = Counter()
generation_mutation_counter = Counter()
total_attack_counter = Counter()  # Contador acumulativo total de ataques
total_mutation_counter = Counter()  # Contador acumulativo total de mutaciones
if USE_RL_WEIGHTS:
    metrics.setdefault("attack_weights_history", []).append(list(attack_q_values))
    metrics.setdefault("mutation_weights_history", []).append(list(mutation_q_values))
else:
    metrics.setdefault("attack_weights_history", []).append(list(attack_weights))
    metrics.setdefault("mutation_weights_history", []).append(list(mutation_weights))

def write_metrics():
    """Persist current metrics to disk"""
    try:
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        # Silently ignore errors when persisting metrics so the fuzzer can
        # continue running even if writing fails (e.g. due to filesystem issues)
        pass

def update_rl_weights():
    """Record current weight values for analysis."""

    if USE_RL_WEIGHTS:
        metrics.setdefault("attack_weights_history", []).append(list(attack_q_values))
        metrics.setdefault("mutation_weights_history", []).append(list(mutation_q_values))
        attack_success.clear()
        mutation_success.clear()
    else:
        metrics.setdefault("attack_weights_history", []).append(list(attack_weights))
        metrics.setdefault("mutation_weights_history", []).append(list(mutation_weights))

    write_metrics()

# Contador para generaciones consecutivas sin crashes por individuo
individual_zero_crash_counts = {}
MAX_INDIVIDUAL_ZERO_CRASH_GENERATIONS = 15  # Máximo de generaciones sin crash para un individuo

def remove_intermediate_exit_syscalls(shellcode):
    """Elimina los EXIT_SYSCALL intermedios del shellcode"""
    if not shellcode:
        return shellcode

    # Primero eliminar EXIT_SYSCALL al final si existe
    if shellcode.endswith(EXIT_SYSCALL):
        core = shellcode[:-len(EXIT_SYSCALL)]
    else:
        core = shellcode

    # Eliminar cualquier EXIT_SYSCALL interno
    i = 0
    while i <= len(core) - len(EXIT_SYSCALL):
        if core[i:i+len(EXIT_SYSCALL)] == EXIT_SYSCALL:
            core = core[:i] + core[i+len(EXIT_SYSCALL):]
        else:
            i += 1

    return core

def deduplicate_population(population):
    """Remove duplicate shellcodes while preserving order"""
    unique = []
    seen = set()
    for indiv in population:
        if indiv not in seen:
            unique.append(indiv)
            seen.add(indiv)
    return unique

def generate_random_instruction():
    """Genera una instrucción aleatoria con mayor probabilidad de instrucciones interesantes"""
    global last_attack_type, total_attack_counter, generation_attack_counter
    if USE_RL_WEIGHTS:
        current_epsilon = get_epsilon(current_generation)
        choice_type = select_with_epsilon_greedy(ATTACK_OPTIONS, attack_q_values, current_epsilon)
    else:
        choice_type = random.choices(ATTACK_OPTIONS, weights=attack_weights)[0]
    last_attack_type = choice_type
    generation_attack_counter[choice_type] += 1
    total_attack_counter[choice_type] += 1  # Nuevo: contador acumulativo
    metrics.setdefault('attack_totals', {})[choice_type] = total_attack_counter[choice_type]
    write_metrics()

    if choice_type == "random_bytes":
        #instr_length = randint(1, 6)
        instr_length = randint(4, 12)
        return bytes([randint(0, 255) for _ in range(instr_length)])

    elif choice_type == "syscall_setup":
        # Generar un número de syscall aleatorio o uno específico interesante
        interesting_syscalls = [
            0,    # read
            1,    # write
            2,    # open
            9,    # mmap
            10,   # mprotect
            60,   # exit
            105,  # setuid
            106,  # setgid
            157,  # prctl
            231,  # exitgroup
        ]

        # 70% probabilidad de usar un syscall específico interesante
        if random.random() < 0.5:
            syscall_num = random.choice(interesting_syscalls)
        else:
            syscall_num = randint(0, 335)  # Número aproximado de syscalls en Linux x86_64

        # Configurar rax con el número de syscall
        setup = b"\x48\xc7\xc0" + syscall_num.to_bytes(4, byteorder='little')

        # Añadir configuración aleatoria para argumentos
        if random.random() < 0.7:
            setup += random.choice(SYSCALL_SETUP[1:])

        return setup

    elif choice_type == "syscall":
        return SYSCALL_PATTERN
        
    elif choice_type == "cache_pollution":
        return generate_cache_pollution_fragment(min_ops=5, max_ops=15)
    
    elif choice_type == "control_flow_trap":
        return generate_control_flow_trap_fragment(min_instr=2, max_instr=6)  
    elif choice_type == "deep_rop_chain":
        return generate_deep_rop_chain_fragment(min_gadgets=4, max_gadgets=10)
    elif choice_type == "syscall_table_stress":
        return generate_syscall_table_stress_fragment(min_calls=5, max_calls=20)    
    elif choice_type == "dma_confusion":
        return generate_dma_confusion_fragment(min_ops=4, max_ops=10)
    elif choice_type == "syscall_storm":
        return generate_syscall_storm(min_calls=5, max_calls=20)   
    elif choice_type == "entropy_drain":
        return generate_entropy_drain_fragment(min_ops=5, max_ops=20)
    elif choice_type == "syscall_reentrancy_storm":
        return generate_syscall_reentrancy_storm_fragment(min_chains=3, max_chains=10)    
    elif choice_type == "external_adn":
        return get_random_fragment_from_bin()
    elif choice_type == "speculative_confusion":
        return generate_speculative_confusion_fragment(min_instr=3, max_instr=8)    
    elif choice_type == "function_adn":
        return get_function_or_fragment()
    elif choice_type == "smap_smep_bypass":
        return generate_smap_smep_bypass_fragment(min_ops=20, max_ops=35)    
    elif choice_type == "filesystem_chaos":
        return generate_filesystem_chaos_fragment(min_ops=5, max_ops=20)
    elif choice_type == "shadow_corruptor":
        return generate_shadow_memory_corruptor_fragment(min_ops=10, max_ops=25)    
    elif choice_type == "gene_bank":
        return get_random_gene()
    elif choice_type == "privileged_storm":
        return generate_privileged_storm_fragment(min_instr=3, max_instr=10)        
    elif choice_type == "gene_bank_dynamic":
        return get_random_gene_dynamic()
    elif choice_type == "privileged_cpu_destruction":
        return generate_privileged_cpu_destruction_fragment(min_instr=2, max_instr=6)        
    elif choice_type == "hyper_corruptor":
        return generate_hyper_advanced_corruptor_fragment(min_ops=15, max_ops=30)
    elif choice_type == "module_loading_storm":
        return generate_module_loading_storm_fragment(min_ops=5, max_ops=15)
    elif choice_type == "pointer_attack":
        return generate_pointer_attack_fragment(min_ops=2, max_ops=6)    
    elif choice_type == "interrupt_storm":
        return generate_interrupt_storm_fragment(min_interrupts=5, max_interrupts=15)        
    elif choice_type == "ipc_stress":
        return generate_ipc_stress_fragment(min_ops=5, max_ops=15)
    elif choice_type == "kpti_breaker":
        return generate_kpti_breaker_fragment(min_ops=15, max_ops=30)        
    elif choice_type == "memory_fragmentation":
        return generate_memory_fragmentation_fragment(min_ops=5, max_ops=15)
    elif choice_type == "network_stack_fuzz":
        return generate_network_stack_fuzz_fragment(min_ops=5, max_ops=15)
    elif choice_type == "neutral_mutation":
        return insert_neutral_mutation(b"", min_insertions=1, max_insertions=3)
    elif choice_type == "nop_island":
        return generate_nop_island(min_nops=2, max_nops=8)  
    elif choice_type == "page_fault_flood":
        return generate_page_fault_flood_fragment(min_faults=5, max_faults=20)   
    elif choice_type == "resource_starvation":
        return generate_resource_starvation_fragment(min_ops=5, max_ops=15)        
    elif choice_type == "scheduler_attack":
        return generate_scheduler_attack_fragment(min_ops=5, max_ops=15)
    elif choice_type == "ultimate_panic":
        return generate_ultimate_panic_fragment(min_ops=20, max_ops=40)
    elif choice_type == "ai_shellcode":
        return generate_ai_shellcode_fragment()
    elif choice_type == "memory_access":
        # Instrucciones que acceden a memoria, más probabilidad de fallos
        mem_instructions = [
            b"\x48\x8b", # mov reg, [reg]
            b"\x48\x89", # mov [reg], reg
            b"\xff",     # varios opcodes que pueden acceder/modificar memoria
            b"\x0f\xae", # instrucciones MFENCE, LFENCE, SFENCE, CLFLUSH
            b"\x48\x8a", # mov reg8, [reg]
            b"\x48\x88", # mov [reg], reg8
            b"\x48\xa4", # movsb
            b"\x48\xa5", # movsq
            b"\x48\xaa", # stosb
            b"\x48\xab", # stosq
        ]
        instr = random.choice(mem_instructions)
        # Añadir bytes aleatorios como operandos
        return instr + bytes([randint(0, 255) for _ in range(randint(1, 3))])
    
    elif choice_type == "memory_pressure":
        return generate_memory_pressure_fragment(min_ops=1, max_ops=3)
    
    elif choice_type == "privileged":
        # Instrucciones privilegiadas o que podrían causar excepciones
        privileged_instructions = [
            b"\x0f\x01", # Varias instrucciones de sistema (SIDT, SGDT, etc.)
            b"\xcd\x80", # int 0x80 (syscall de 32-bit)
            b"\xf4",     # HLT (halt)
            b"\x0f\x00", # Instrucciones de control de segmentos
            b"\x0f\x06", # CLTS (clear task switched flag)
            b"\x0f\x09", # WBINVD (write back and invalidate cache)
            b"\x0f\x30", # WRMSR (write to model specific register)
            b"\x0f\x32", # RDMSR (read from model specific register)
            b"\x0f\x34", # SYSENTER
            b"\x0f\x35", # SYSEXIT
            b"\x0f\x07", # SYSRET
            b"\x0f\x05", # SYSCALL
            b"\x0f\x0b", # UD2 (undefined instruction)
        ]
        instr = random.choice(privileged_instructions)
        # Algunos necesitan operandos
        if instr in [b"\x0f\x01", b"\x0f\x00"]:
            return instr + bytes([randint(0, 255) for _ in range(1)])
        return instr

    elif choice_type == "known_vulns":
        # Bloques derivados de vulnerabilidades conocidas en kernels Linux
        vuln_instruction_blocks = [
            b"\x0f\xae\x05",          # Vulnerabilidad CVE-2018-8897 (pop SS, mov a registro segmentado)
            b"\x65\x48\x8b\x04\x25",  # Lectura arbitraria usando GS register (usado en exploits recientes)
            b"\x0f\x3f",              # Instrucción AMD específica que generó excepciones en Intel CPUs
            b"\xf3\x0f\xae\xf0",      # Especulativa ejecución (Spectre/Meltdown-type gadgets)
            b"\x48\xcf",              # IRETQ en contexto mal configurado (CVE-2014-9322)
            b"\x0f\x22\xc0",          # MOV to CR0 (Control register manipulation, usado en exploits antiguos)
            b"\x0f\x32",              # RDMSR (lectura modelo-específica de registros usada en varios exploits)
            b"\x0f\x30",              # WRMSR (escritura modelo-específica, potencial corrupción)
            b"\x0f\x01\xd0",          # XGETBV (registro de estado extendido, CVE potencial en CPUs antiguas)
            b"\x0f\x01\xf8",          # SWAPGS (usado en Spectre v1, v4, otros CVEs relacionados)
            b"\x0f\xae\x38",          # CLFLUSH (Spectre-style caché side-channel)
            b"\x0f\x18",              # PREFETCH (utilizado en algunos side-channels)
        ]
        instr = random.choice(vuln_instruction_blocks)
        # Algunos bloques podrían necesitar bytes adicionales
        if instr in [b"\x65\x48\x8b\x04\x25", b"\x0f\xae\x05", b"\x0f\x22\xc0"]:
            instr += bytes([randint(0, 255) for _ in range(2)])
        return instr

    elif choice_type == "arithmetic":
        # Operaciones aritméticas que podrían causar excepciones
        arith_instructions = [
            b"\x48\x01", # add r/m64, r64
            b"\x48\x29", # sub r/m64, r64
            b"\x48\xf7", # mul/div/not/neg r/m64
            b"\x48\x0f\xaf", # imul r64, r/m64
            b"\x48\x0f\xc7", # cmpxchg8b/16b
            b"\x48\x99", # cqo (sign-extend)
            b"\x48\xd1", # rol/ror/rcl/rcr/shl/shr/sar r/m64, 1
        ]
        instr = random.choice(arith_instructions)
        # Añadir operandos aleatorios
        op_len = 1 if len(instr) > 2 else randint(1, 2)
        return instr + bytes([randint(0, 255) for _ in range(op_len)])

    elif choice_type == "full_kernel_syscall":
        #Todas las funciones del kernel
        syscall_num = random.choice(list(syscalls.keys()))
        
        # Construir syscall ASM con número aleatorio
        setup = b"\x48\xc7\xc0" + syscall_num.to_bytes(4, byteorder='little')  # mov rax, syscall_num
        # argumentos aleatorios simples (probablemente ceros o pequeños enteros)
        for reg in [b"\x48\x31\xff", b"\x48\x31\xf6", b"\x48\x31\xd2"]:  # rdi, rsi, rdx
            if random.random() < 0.7:
                setup += reg
        setup += SYSCALL_PATTERN  # ejecutar syscall
        return setup

    elif choice_type == "control_flow":
        # Instrucciones de salto que podrían causar comportamientos interesantes
        cf_instructions = [
            b"\xe9", # jmp rel32
            b"\xeb", # jmp rel8
            b"\x74", # je/jz rel8
            b"\x75", # jne/jnz rel8
            b"\x0f\x84", # je/jz rel32
            b"\x0f\x85", # jne/jnz rel32
            b"\xe8", # call rel32
            b"\xff", # call/jmp r/m64
            b"\xc3", # ret
            b"\xc2", # ret imm16
        ]
        instr = random.choice(cf_instructions)
        # Añadir operandos según el tipo de instrucción
        if instr == b"\xe9" or instr == b"\xe8":
            # jmp/call rel32 (4 bytes de desplazamiento)
            return instr + bytes([randint(0, 255) for _ in range(4)])
        elif instr == b"\xeb" or instr == b"\x74" or instr == b"\x75":
            # jmp/jz/jnz rel8 (1 byte de desplazamiento)
            return instr + bytes([randint(0, 255)])
        elif instr == b"\x0f\x84" or instr == b"\x0f\x85":
            # jz/jnz rel32 (4 bytes de desplazamiento)
            return instr + bytes([randint(0, 255) for _ in range(4)])
        elif instr == b"\xc2":
            # ret imm16 (2 bytes inmediatos)
            return instr + bytes([randint(0, 255) for _ in range(2)])
        elif instr == b"\xff":
            # call/jmp r/m64 (necesita modrm)
            return instr + bytes([randint(0, 255)])
        else:
            return instr

    elif choice_type == "simd":
        # Instrucciones SIMD (SSE, AVX) que podrían causar excepciones
        simd_prefixes = [
            b"\x66", # Prefijo operando 16-bit
            b"\xf2", # Prefijo REPNE/SSE
            b"\xf3", # Prefijo REP/SSE
        ]

        simd_opcodes = [
            b"\x0f\x10", # movups/movss xmm, xmm/m
            b"\x0f\x11", # movups/movss xmm/m, xmm
            b"\x0f\x28", # movaps xmm, xmm/m
            b"\x0f\x29", # movaps xmm/m, xmm
            b"\x0f\x58", # addps/addss xmm, xmm/m
            b"\x0f\x59", # mulps/mulss xmm, xmm/m
            b"\x0f\x6f", # movq mm, mm/m
            b"\x0f\x7f", # movq mm/m, mm
            b"\x0f\xae", # [diversos, según modrm]
            b"\x0f\xc2", # cmpps/cmpss xmm, xmm/m, imm8
        ]

        # 50% de chance de usar un prefijo
        if random.random() < 0.5:
            prefix = random.choice(simd_prefixes)
            opcode = random.choice(simd_opcodes)
            modrm = bytes([randint(0, 255)])
            # Algunos requieren un byte inmediato adicional
            if opcode == b"\x0f\xc2":
                return prefix + opcode + modrm + bytes([randint(0, 7)])
            return prefix + opcode + modrm
        else:
            opcode = random.choice(simd_opcodes)
            modrm = bytes([randint(0, 255)])
            # Algunos requieren un byte inmediato adicional
            if opcode == b"\x0f\xc2":
                return opcode + modrm + bytes([randint(0, 7)])
            return opcode + modrm
    elif choice_type == "segment_registers":
        segment_instr_blocks = [
            b"\x8e\xd8",       # mov ds, ax
            b"\x8e\xc0",       # mov es, ax
            b"\x8e\xe0",       # mov fs, ax
            b"\x8e\xe8",       # mov gs, ax
            b"\x8c\xd8",       # mov ax, ds
            b"\x8c\xc0",       # mov ax, es
            b"\x8c\xe0",       # mov ax, fs
            b"\x8c\xe8",       # mov ax, gs
        ]
        return random.choice(segment_instr_blocks)

    elif choice_type == "speculative_exec":
        speculative_blocks = [
            b"\x0f\xae\x38",  # CLFLUSH (flush cache)
            b"\x0f\xae\xf8",  # SFENCE (store fence)
            b"\x0f\xae\xe8",  # LFENCE (load fence)
            b"\x0f\xae\xf0",  # MFENCE (memory fence)
            b"\x0f\x31",      # RDTSC (leer contador timestamp)
            b"\x0f\xc7\xf8",  # RDPID (leer Process ID)
        ]
        return random.choice(speculative_blocks)

    elif choice_type == "forced_exception":
        exception_blocks = [
            b"\x0f\x0b",      # UD2 (Undefined instruction)
            b"\xcc",          # INT3 (Breakpoint)
            b"\xcd\x03",      # INT 3 (Alternate breakpoint)
            b"\xcd\x04",      # INTO (Overflow exception)
            b"\xf4",          # HLT (Halt instruction)
            b"\xce",          # INTO (Overflow - condición en flags)
        ]
        return random.choice(exception_blocks)

    elif choice_type == "control_registers":
        control_register_blocks = [
            b"\x0f\x22\xd8",  # MOV CR3, RAX (set page directory)
            b"\x0f\x20\xd8",  # MOV RAX, CR3 (leer page directory)
            b"\x0f\x22\xe0",  # MOV CR4, RAX (set control flags)
            b"\x0f\x20\xe0",  # MOV RAX, CR4 (leer control flags)
            b"\x0f\x22\xd0",  # MOV CR2, RAX (set page fault address)
            b"\x0f\x20\xd0",  # MOV RAX, CR2 (leer page fault address)
        ]
        return random.choice(control_register_blocks)
    elif choice_type == "stack_manipulation":
        stack_blocks = [
            b"\x48\x89\xe4",   # mov rsp, rbp
            b"\x48\x83\xc4",   # add rsp, imm8
            b"\x48\x83\xec",   # sub rsp, imm8
            b"\x48\x8d\x64\x24",  # lea rsp, [rsp+disp8]
            b"\x9c",           # pushfq (guardar flags en stack)
            b"\x9d",           # popfq (cargar flags desde stack)
        ]
        instr = random.choice(stack_blocks)
        if instr in [b"\x48\x83\xc4", b"\x48\x83\xec", b"\x48\x8d\x64\x24"]:
            instr += bytes([randint(0, 255)])
        return instr

    elif choice_type == "x86_opcode":
        # Generar un opcode x86 completamente aleatorio
        # Comenzar con prefijos (opcionales)
        prefixes = [
            b"", # sin prefijo (más común)
            b"\x66", # prefijo operando 16-bit
            b"\x67", # prefijo dirección 32-bit
            b"\xf0", # prefijo LOCK
            b"\xf2", # prefijo REPNE
            b"\xf3", # prefijo REP
            b"\x2e", # prefijo CS
            b"\x36", # prefijo SS
            b"\x3e", # prefijo DS
            b"\x26", # prefijo ES
            b"\x64", # prefijo FS
            b"\x65", # prefijo GS
        ]

        # Prefijos REX (para instrucciones de 64 bits)
        rex_prefixes = [
            b"", # sin REX (más común)
            b"\x40", # REX
            b"\x41", # REX.B
            b"\x42", # REX.X
            b"\x43", # REX.XB
            b"\x44", # REX.R
            b"\x45", # REX.RB
            b"\x46", # REX.RX
            b"\x47", # REX.RXB
            b"\x48", # REX.W
            b"\x49", # REX.WB
            b"\x4a", # REX.WX
            b"\x4b", # REX.WXB
            b"\x4c", # REX.WR
            b"\x4d", # REX.WRB
            b"\x4e", # REX.WRX
            b"\x4f", # REX.WRXB
        ]

        prefix = random.choice(prefixes)
        rex = random.choice(rex_prefixes)

        # Mayor probabilidad de opcode de 1 byte
        if random.random() < 0.7:
            # Opcode de 1 byte (excluyendo algunos rangos reservados para escape de 2 bytes)
            while True:
                opcode = bytes([randint(0, 255)])
                # Evitar los bytes de escape multi-byte 0x0F, 0xD8-0xDF (instrucciones x87)
                if opcode[0] != 0x0F and not (0xD8 <= opcode[0] <= 0xDF):
                    break
        else:
            # Opcode de 2 bytes (comenzando con 0x0F)
            opcode = b"\x0f" + bytes([randint(0, 255)])

        # ModR/M byte (probabilidad alta)
        if random.random() < 0.8:
            modrm = bytes([randint(0, 255)])
        else:
            modrm = b""

        # SIB byte (probabilidad baja, solo si ModR/M lo requiere)
        if modrm and modrm[0] & 0b11000111 == 0b00000100:
            sib = bytes([randint(0, 255)])
        else:
            sib = b""

        # Desplazamiento (probabilidad media, dependiendo de ModR/M)
        if modrm:
            mod = modrm[0] >> 6
            rm = modrm[0] & 0b00000111

            if mod == 0b01:
                # Desplazamiento de 1 byte
                disp = bytes([randint(0, 255)])
            elif mod == 0b10 or (mod == 0b00 and rm == 0b101):
                # Desplazamiento de 4 bytes
                disp = bytes([randint(0, 255) for _ in range(4)])
            else:
                disp = b""
        else:
            disp = b""

        # Inmediato (probabilidad baja)
        if random.random() < 0.3:
            imm_size = random.choices([1, 2, 4], weights=[60, 20, 20])[0]
            imm = bytes([randint(0, 255) for _ in range(imm_size)])
        else:
            imm = b""

        # Construir la instrucción completa
        instruction = prefix + rex + opcode + modrm + sib + disp + imm

        # Limitar el tamaño máximo para que sea una instrucción razonable
        return instruction[:min(len(instruction), 15)]

    # Por defecto
    return bytes([randint(0, 255) for _ in range(randint(1, 4))])

def mutate_shellcode(shellcode, mutation_rate=0.8):
    """Muta el shellcode con diferentes estrategias"""
    # Eliminar EXIT_SYSCALL si existe al final
    if shellcode.endswith(EXIT_SYSCALL):
        core = shellcode[:-len(EXIT_SYSCALL)]
    else:
        core = shellcode

    # Asegurar que no hay EXIT_SYSCALL intermedios
    core = remove_intermediate_exit_syscalls(core)

    global last_mutation_type, total_mutation_counter, generation_mutation_counter
    if USE_RL_WEIGHTS:
        current_epsilon = get_epsilon(current_generation)
        mutation_type = select_with_epsilon_greedy(MUTATION_TYPES, mutation_q_values, current_epsilon)
    else:
        mutation_type = random.choices(MUTATION_TYPES, weights=mutation_weights)[0]
    last_mutation_type = mutation_type
    generation_mutation_counter[mutation_type] += 1
    total_mutation_counter[mutation_type] += 1  # Nuevo: contador acumulativo
    metrics.setdefault("mutation_totals", {})[mutation_type] = total_mutation_counter[mutation_type]
    write_metrics()

    if mutation_type == "add" or not core:
        # Añadir instrucción (caso más común)
        instr = generate_random_instruction()
        insert_pos = randint(0, max(0, len(core) - 1)) if core else 0
        new_core = core[:insert_pos] + instr + core[insert_pos:]

    elif mutation_type == "remove" and len(core) > 4:
        # Eliminar algunos bytes
        remove_start = randint(0, len(core) - 4)
        remove_length = randint(1, min(4, len(core) - remove_start))
        new_core = core[:remove_start] + core[remove_start + remove_length:]
    
    elif mutation_type == "invert" and len(core) >= 4:
        # Invertir un fragmento aleatorio del shellcode
        new_core = invert_fragment(core)
        
    elif mutation_type == "transpose_nop" and len(core) >= 6:
        # Transponer fragmentos respetando islas de NOP
        new_core = transpose_fragment_nop_aware(core)
        
    elif mutation_type == "modify" and core:
        # Modificar un byte existente
        mod_pos = randint(0, len(core) - 1)
        mod_length = randint(1, min(4, len(core) - mod_pos))
        new_core = (core[:mod_pos] +
                   bytes([randint(0, 255) for _ in range(mod_length)]) +
                   core[mod_pos + mod_length:])

    elif mutation_type == "duplicate" and len(core) >= 4:
        # Duplicar una sección
        dup_start = randint(0, len(core) - 4)
        dup_length = randint(2, min(8, len(core) - dup_start))
        dup_section = core[dup_start:dup_start + dup_length]
        insert_pos = randint(0, len(core))
        new_core = core[:insert_pos] + dup_section + core[insert_pos:]
    
    elif mutation_type == "mass_duplicate" and len(core) >= 4:
        # Usar rutina avanzada de duplicación
        new_core = duplicate_fragment(core)
    
    elif mutation_type == "crispr" and core:
        # Edición específica de syscalls mediante CRISPR
        new_core = crispr_edit_shellcode(core)
        
    else:
        # Si no se pudo aplicar la mutación seleccionada
        instr = generate_random_instruction()
        new_core = core + instr

    # 20% de probabilidad de no añadir EXIT_SYSCALL (más agresivo)
    if random.random() < 0.2:
        return new_core
    else:
        return new_core + EXIT_SYSCALL

def is_system_level_crash(return_code, stderr):
    """Identifica si un crash puede haber afectado al sistema operativo"""
    # Señales que podrían indicar impacto a nivel de sistema
    #system_level_signals = [
    #    signal.SIGBUS,
    #    signal.SIGSYS,
    #    signal.SIGILL,
    #    signal.SIGABRT,
    #    signal.SIGSEGV,
    #    signal.SIGFPE,
    #    signal.SIGTRAP,
    #]
    system_level_signals = [
        sig for sig in signal.Signals
    ]
    if return_code < 0 and -return_code in [s.value for s in system_level_signals]:
        return True

    # Buscar mensajes de error del kernel en stderr
    kernel_indicators = ["kernel", "panic", "oops", "BUG:", "WARNING:", "general protection"]
    if any(indicator in stderr.lower() for indicator in kernel_indicators):
        return True

    return False

def generate_c_stub(shellcode_c):
    """Return the modular C stub used to execute the shellcode"""
    return f"""
            #include <stdio.h>
            #include <stdlib.h>
            #include <sys/mman.h>
            #include <string.h>
            #include <unistd.h>
            #include <sys/socket.h>
            #include <netinet/in.h>
            #include <arpa/inet.h>

            #define FUNC_SOCKET    500
            #define FUNC_CONNECT   501
            #define FUNC_LISTEN    502
            #define FUNC_SEND      503
            #define FUNC_RECV      504
            #define FUNC_EXECUTE   505

            unsigned char code[] = {{{shellcode_c}}};

            // Estructura para almacenar información de conexión
            struct connection_info {{
                int sockfd;
                struct sockaddr_in addr;
                int is_connected;
            }} conn_info = {{-1, {{0}}, 0}};

            int open_socket() {{
                int sockfd = socket(AF_INET, SOCK_STREAM, 0);
                if (sockfd >= 0) {{
                    write(1, "[+] Socket opened\\n", 18);
                    conn_info.sockfd = sockfd;
                }} else {{
                    write(1, "[-] Failed to open socket\\n", 26);
                }}
                return sockfd;
            }}

            int connect_socket(const char *ip, int port) {{
                if (conn_info.sockfd < 0) {{
                    write(1, "[-] No socket available\\n", 24);
                    return -1;
                }}

                conn_info.addr.sin_family = AF_INET;
                conn_info.addr.sin_port = htons(port);
                conn_info.addr.sin_addr.s_addr = inet_addr(ip);

                if (connect(conn_info.sockfd, (struct sockaddr *)&conn_info.addr, sizeof(conn_info.addr)) == 0) {{
                    write(1, "[+] Connected\\n", 14);
                    conn_info.is_connected = 1;
                    return 0;
                }} else {{
                    write(1, "[-] Connection failed\\n", 22);
                    return -1;
                }}
            }}

            int listen_socket(int port) {{
                if (conn_info.sockfd < 0) {{
                    write(1, "[-] No socket available\\n", 24);
                    return -1;
                }}

                conn_info.addr.sin_family = AF_INET;
                conn_info.addr.sin_port = htons(port);
                conn_info.addr.sin_addr.s_addr = INADDR_ANY;

                if (bind(conn_info.sockfd, (struct sockaddr *)&conn_info.addr, sizeof(conn_info.addr)) < 0) {{
                    write(1, "[-] Bind failed\\n", 16);
                    return -1;
                }}

                if (listen(conn_info.sockfd, 5) < 0) {{
                    write(1, "[-] Listen failed\\n", 18);
                    return -1;
                }}

                write(1, "[+] Listening\\n", 14);
                return 0;
            }}

            int send_data(const char *data, size_t len) {{
                if (conn_info.sockfd < 0 || !conn_info.is_connected) {{
                    write(1, "[-] Not connected\\n", 18);
                    return -1;
                }}

                ssize_t bytes_sent = send(conn_info.sockfd, data, len, 0);
                if (bytes_sent > 0) {{
                    write(1, "[+] Data sent\\n", 14);
                }} else {{
                    write(1, "[-] Send failed\\n", 16);
                }}
                return bytes_sent;
            }}

            int recv_data(char *buffer, size_t len) {{
                if (conn_info.sockfd < 0 || !conn_info.is_connected) {{
                    write(1, "[-] Not connected\\n", 18);
                    return -1;
                }}

                ssize_t bytes_recv = recv(conn_info.sockfd, buffer, len, 0);
                if (bytes_recv > 0) {{
                    write(1, "[+] Data received\\n", 18);
                }} else {{
                    write(1, "[-] Receive failed\\n", 19);
                }}
                return bytes_recv;
            }}

            void execute_payload(const char *payload, size_t len) {{
                void *mem = mmap(0, len, PROT_READ | PROT_WRITE | PROT_EXEC,
                                MAP_ANON | MAP_PRIVATE, -1, 0);
                if (mem == MAP_FAILED) {{
                    write(1, "[-] Memory allocation failed\\n", 29);
                    return;
                }}

                memcpy(mem, payload, len);
                write(1, "[+] Executing payload\\n", 22);
                ((void(*)())mem)();
                munmap(mem, len);
            }}

            void dispatch(int action) {{
                switch (action) {{
                    case FUNC_SOCKET:
                        open_socket();
                        break;
                    case FUNC_CONNECT:
                        write(1, "[*] Connecting to 127.0.0.1:4444\\n", 33);
                        connect_socket("127.0.0.1", 4444);
                        break;
                    case FUNC_LISTEN:
                        write(1, "[*] Listening on port 4444\\n", 27);
                        listen_socket(4444);
                        break;
                    case FUNC_SEND:
                        write(1, "[*] Sending data...\\n", 20);
                        send_data("Hello from stub\\n", 16);
                        break;
                    case FUNC_RECV: {{
                        write(1, "[*] Receiving data...\\n", 22);
                        char buffer[1024];
                        recv_data(buffer, sizeof(buffer));
                        break;
                    }}
                    case FUNC_EXECUTE:
                        write(1, "[*] Executing payload...\\n", 25);
                        execute_payload((char *)code, sizeof(code));
                        break;
                    default:
                        write(1, "[-] Unknown action.\\n", 21);
                        break;
                }}
            }}

            int main() {{
                setbuf(stdout, NULL);
                setbuf(stderr, NULL);

                void *exec = mmap(0, sizeof(code), PROT_READ | PROT_WRITE | PROT_EXEC,
                                MAP_ANON | MAP_PRIVATE, -1, 0);

                if (exec == MAP_FAILED) {{
                    perror("mmap");
                    return 1;
                }}

                memcpy(exec, code, sizeof(code));

                // Ejecutar el shellcode
                ((void(*)())exec)();

                // El shellcode establece rdi = 500, verificar si necesitamos dispatch
                register int action asm("rdi");
                if (action >= 500) {{
                    dispatch(action);
                }}

                // Limpiar recursos
                if (conn_info.sockfd >= 0) {{
                    close(conn_info.sockfd);
                }}
                munmap(exec, sizeof(code));

                return 0;
            }}
            """

def compile_and_execute_worker(data):
    """Compile the C program and execute the resulting binary."""
    c_path = data["c_path"]
    bin_path = data["bin_path"]
    timeout = data["timeout"]
    try:
        subprocess.run(["clang", c_path, "-o", bin_path], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return {"status": "COMPILE_ERROR", "result": None}

    try:
        result = subprocess.run([bin_path], timeout=timeout, capture_output=True)
        return {"status": "OK", "result": result}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "result": None}
    except Exception as e:
        return {"status": "ERROR", "error": str(e), "result": None}

async def process_single_program_async(executor, program_id, parent_shellcode, gen_id, gen_path):
    """Procesa un programa individual de forma asíncrona"""
    shellcode = mutate_shellcode(parent_shellcode)
    shellcode_c = format_shellcode_c_array(shellcode)
    stub = generate_c_stub(shellcode_c)

    c_path = os.path.join(gen_path, f"g{gen_id:04d}_p{program_id:04d}.c")
    bin_path = os.path.join(gen_path, f"g{gen_id:04d}_p{program_id:04d}")

    with open(c_path, "w") as f:
        f.write(stub)

    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(executor, compile_and_execute_worker,
                                     {"c_path": c_path, "bin_path": bin_path, "timeout": TIMEOUT})

    survived = False
    crash_info = None
    result_obj = res.get("result")
    if res["status"] == "OK":
        survived = result_obj.returncode == 0
        if not survived:
            stderr_text = result_obj.stderr.decode('utf-8', errors='replace')
            stdout_text = result_obj.stdout.decode('utf-8', errors='replace')
            is_system = is_system_level_crash(result_obj.returncode, stderr_text)
            if result_obj.returncode < 0:
                sig_value = -result_obj.returncode
                try:
                    sig_name = signal.Signals(sig_value).name
                except ValueError:
                    sig_name = f"UNKNOWN_{sig_value}"
                crash_type = f"SIGNAL_{sig_name}"
            else:
                crash_type = f"EXIT_{result_obj.returncode}"
            crash_info = {
                "crash_type": crash_type,
                "return_code": result_obj.returncode,
                "stderr": stderr_text,
                "stdout": stdout_text,
                "system_impact": is_system,
                "parent": parent_shellcode,
            }
    else:
        crash_info = {"crash_type": res["status"], "return_code": None}

    return {
        "program_id": program_id,
        "shellcode": shellcode,
        "result": result_obj if res["status"] == "OK" else res["status"],
        "crash_info": crash_info,
        "survived": survived,
        "stub": stub,
    }


#def save_crash_info(gen_id, prog_id, shellcode, crash_type, stderr, stdout, return_code, is_system_impact):
def save_crash_info(gen_id, prog_id, shellcode, crash_type, stderr, stdout, return_code, is_system_impact, parent_shellcode=None):    
    """Guarda información detallada del crash para análisis posterior"""
    # Determinar directorio según impacto
    save_dir = "kernelhunter_critical" if is_system_impact else "kernelhunter_crashes"

    # Crear archivo con información detallada
    crash_info = {
        "generation": gen_id,
        "program_id": prog_id,
        "shellcode_hex": shellcode.hex(),
        "shellcode_length": len(shellcode),
        "crash_type": crash_type,
        "return_code": return_code,
        "stderr": stderr,
        "stdout": stdout,
        "system_impact": is_system_impact,
        "timestamp": time.time(),
        "parent_shellcode_hex": parent_shellcode.hex() if parent_shellcode else None
    }

    with open(f"{save_dir}/crash_gen{gen_id:04d}_prog{prog_id:04d}.json", "w") as f:
        json.dump(crash_info, f, indent=2)

def run_generation(gen_id, base_population):
    """Ejecuta una generación completa del algoritmo evolutivo"""
    global individual_zero_crash_counts, generation_attack_counter, generation_mutation_counter
    global attack_success, mutation_success, current_generation
    current_generation = gen_id

    # Reset counters for this generation
    generation_attack_counter = Counter()
    generation_mutation_counter = Counter()
    attack_success.clear()
    mutation_success.clear()

    gen_path = os.path.join(OUTPUT_DIR, f"gen_{gen_id:04d}")
    os.makedirs(gen_path, exist_ok=True)

    new_population = []
    crashes = 0
    crash_types_counter = Counter()
    system_impacts = 0
    shellcode_lengths = []

    with open(LOG_FILE, "a") as log, open(CRASH_LOG, "a") as crash_log:
        log.write(f"\n[GEN {gen_id}] {'='*50}\n")
        crash_log.write(f"\n[GEN {gen_id}] {'='*50}\n")

        for i in range(NUM_PROGRAMS):
            if i % 5 == 0:  # Mostrar progreso cada 5 programas
                print(f"Generando individuos {i}/{NUM_PROGRAMS}", end="\r")

            # Selección natural pura - sin sesgos artificiales
            parent = choice(base_population)

            if random.random() < 0.3:
                other_parent = choice(base_population)
                shellcode = crossover_shellcodes_advanced(parent, other_parent)
            else:
                shellcode = mutate_shellcode(parent)
            
            shellcode_c = format_shellcode_c_array(shellcode)
            shellcode_lengths.append(len(shellcode))

            # Generar el programa C con el shellcode
            # Generar el programa C con el shellcode
            # Stub modular en C
            stub = f"""
            #include <stdio.h>
            #include <stdlib.h>
            #include <sys/mman.h>
            #include <string.h>
            #include <unistd.h>
            #include <sys/socket.h>
            #include <netinet/in.h>
            #include <arpa/inet.h>

            #define FUNC_SOCKET    500
            #define FUNC_CONNECT   501
            #define FUNC_LISTEN    502
            #define FUNC_SEND      503
            #define FUNC_RECV      504
            #define FUNC_EXECUTE   505

            unsigned char code[] = {{{shellcode_c}}};

            // Estructura para almacenar información de conexión
            struct connection_info {{
                int sockfd;
                struct sockaddr_in addr;
                int is_connected;
            }} conn_info = {{-1, {{0}}, 0}};

            int open_socket() {{
                int sockfd = socket(AF_INET, SOCK_STREAM, 0);
                if (sockfd >= 0) {{
                    write(1, "[+] Socket opened\\n", 18);
                    conn_info.sockfd = sockfd;
                }} else {{
                    write(1, "[-] Failed to open socket\\n", 26);
                }}
                return sockfd;
            }}

            int connect_socket(const char *ip, int port) {{
                if (conn_info.sockfd < 0) {{
                    write(1, "[-] No socket available\\n", 24);
                    return -1;
                }}
                
                conn_info.addr.sin_family = AF_INET;
                conn_info.addr.sin_port = htons(port);
                conn_info.addr.sin_addr.s_addr = inet_addr(ip);
                
                if (connect(conn_info.sockfd, (struct sockaddr *)&conn_info.addr, sizeof(conn_info.addr)) == 0) {{
                    write(1, "[+] Connected\\n", 14);
                    conn_info.is_connected = 1;
                    return 0;
                }} else {{
                    write(1, "[-] Connection failed\\n", 22);
                    return -1;
                }}
            }}

            int listen_socket(int port) {{
                if (conn_info.sockfd < 0) {{
                    write(1, "[-] No socket available\\n", 24);
                    return -1;
                }}
                
                conn_info.addr.sin_family = AF_INET;
                conn_info.addr.sin_port = htons(port);
                conn_info.addr.sin_addr.s_addr = INADDR_ANY;
                
                if (bind(conn_info.sockfd, (struct sockaddr *)&conn_info.addr, sizeof(conn_info.addr)) < 0) {{
                    write(1, "[-] Bind failed\\n", 16);
                    return -1;
                }}
                
                if (listen(conn_info.sockfd, 5) < 0) {{
                    write(1, "[-] Listen failed\\n", 18);
                    return -1;
                }}
                
                write(1, "[+] Listening\\n", 14);
                return 0;
            }}

            int send_data(const char *data, size_t len) {{
                if (conn_info.sockfd < 0 || !conn_info.is_connected) {{
                    write(1, "[-] Not connected\\n", 18);
                    return -1;
                }}
                
                ssize_t bytes_sent = send(conn_info.sockfd, data, len, 0);
                if (bytes_sent > 0) {{
                    write(1, "[+] Data sent\\n", 14);
                }} else {{
                    write(1, "[-] Send failed\\n", 16);
                }}
                return bytes_sent;
            }}

            int recv_data(char *buffer, size_t len) {{
                if (conn_info.sockfd < 0 || !conn_info.is_connected) {{
                    write(1, "[-] Not connected\\n", 18);
                    return -1;
                }}
                
                ssize_t bytes_recv = recv(conn_info.sockfd, buffer, len, 0);
                if (bytes_recv > 0) {{
                    write(1, "[+] Data received\\n", 18);
                }} else {{
                    write(1, "[-] Receive failed\\n", 19);
                }}
                return bytes_recv;
            }}

            void execute_payload(const char *payload, size_t len) {{
                void *mem = mmap(0, len, PROT_READ | PROT_WRITE | PROT_EXEC,
                                MAP_ANON | MAP_PRIVATE, -1, 0);
                if (mem == MAP_FAILED) {{
                    write(1, "[-] Memory allocation failed\\n", 29);
                    return;
                }}
                
                memcpy(mem, payload, len);
                write(1, "[+] Executing payload\\n", 22);
                ((void(*)())mem)();
                munmap(mem, len);
            }}

            void dispatch(int action) {{
                switch (action) {{
                    case FUNC_SOCKET:
                        open_socket();
                        break;
                    case FUNC_CONNECT:
                        write(1, "[*] Connecting to 127.0.0.1:4444\\n", 33);
                        connect_socket("127.0.0.1", 4444);
                        break;
                    case FUNC_LISTEN:
                        write(1, "[*] Listening on port 4444\\n", 27);
                        listen_socket(4444);
                        break;
                    case FUNC_SEND:
                        write(1, "[*] Sending data...\\n", 20);
                        send_data("Hello from stub\\n", 16);
                        break;
                    case FUNC_RECV: {{
                        write(1, "[*] Receiving data...\\n", 22);
                        char buffer[1024];
                        recv_data(buffer, sizeof(buffer));
                        break;
                    }}
                    case FUNC_EXECUTE:
                        write(1, "[*] Executing payload...\\n", 25);
                        execute_payload((char *)code, sizeof(code));
                        break;
                    default:
                        write(1, "[-] Unknown action.\\n", 21);
                        break;
                }}
            }}

            int main() {{
                setbuf(stdout, NULL);
                setbuf(stderr, NULL);
                
                void *exec = mmap(0, sizeof(code), PROT_READ | PROT_WRITE | PROT_EXEC,
                                MAP_ANON | MAP_PRIVATE, -1, 0);
                
                if (exec == MAP_FAILED) {{
                    perror("mmap");
                    return 1;
                }}
                
                memcpy(exec, code, sizeof(code));
                
                // Ejecutar el shellcode
                ((void(*)())exec)();
                
                // El shellcode establece rdi = 500, verificar si necesitamos dispatch
                register int action asm("rdi");
                if (action >= 500) {{
                    dispatch(action);
                }}
                
                // Limpiar recursos
                if (conn_info.sockfd >= 0) {{
                    close(conn_info.sockfd);
                }}
                munmap(exec, sizeof(code));
                
                return 0;
            }}
            """


            #c_path = os.path.join(gen_path, f"prog_{i:04d}.c")
            #bin_path = os.path.join(gen_path, f"prog_{i:04d}")
            c_path = os.path.join(gen_path, f"g{gen_id:04d}_p{i:04d}.c")
            bin_path = os.path.join(gen_path, f"g{gen_id:04d}_p{i:04d}")
            #c_path = os.path.join(gen_path, f"gen{gen_id:04d}_prog{i:04d}.c")
            #bin_path = os.path.join(gen_path, f"gen{gen_id:04d}_prog{i:04d}")
            
            with open(c_path, "w") as f:
                f.write(stub)

            try:
                # Compilar el programa
                subprocess.run(["clang", c_path, "-o", bin_path], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Ejecutar con timeout
                result = subprocess.run([bin_path], timeout=TIMEOUT, capture_output=True)
                stderr_text = result.stderr.decode('utf-8', errors='replace')
                stdout_text = result.stdout.decode('utf-8', errors='replace')

                reward = 0
                if result.returncode == 0:
                    # El shellcode sobrevivió
                    new_population.append(shellcode)
                    shellcode_hex = print_shellcode_hex(shellcode, escape_format=True)
                    log.write(f"Survivor {i:04d}: {shellcode_hex} (len: {len(shellcode)})\n")
                    if i < 2:  # Solo mostrar algunos ejemplos de sobrevivientes para no saturar la salida
                        print(color_text(f"Survivor sample {i:04d}: {shellcode_hex}", "green"))
                    if USE_RL_WEIGHTS:
                        update_q_value(attack_q_values, attack_counts, attack_index[last_attack_type], 0)
                        update_q_value(mutation_q_values, mutation_counts, mutation_index[last_mutation_type], 0)
                        update_rl_weights()
                else:
                    # Determinar si el crash puede haber afectado al sistema
                    is_system_impact = is_system_level_crash(result.returncode, stderr_text)

                    if result.returncode < 0:
                        # Crash con señal
                        sig_value = -result.returncode
                        try:
                            sig_name = signal.Signals(sig_value).name
                        except ValueError:
                            sig_name = f"UNKNOWN_{sig_value}"

                        crash_type = f"SIGNAL_{sig_name}"
                        shellcode_hex = print_shellcode_hex(shellcode, escape_format=True)
                        msg = f"Crash {i:04d}: {sig_name} | Shellcode: {shellcode_hex}"

                        # Registrar el tipo de crash
                        crash_types_counter[crash_type] += 1

                    else:
                        # Salida con código de error
                        crash_type = f"EXIT_{result.returncode}"
                        shellcode_hex = print_shellcode_hex(shellcode, escape_format=True)
                        msg = f"Crash {i:04d}: exit code {result.returncode} | Shellcode: {shellcode_hex}"
                        crash_types_counter[crash_type] += 1

                    # Marcar impactos del sistema
                    if is_system_impact:
                        msg = f"[SYSTEM IMPACT] {msg}"
                        system_impacts += 1
                        crash_info = {
                            "crash_type": crash_type,
                            "return_code": result.returncode,
                            "child_crashed": True
                        }
                        genetic_reservoir.add(parent, crash_info)  # Save the parent, not the crashed shellcode

                    if is_system_impact:
                        print(color_text(msg, "magenta"))
                    else:
                        print(color_text(msg, "red"))
                    crash_log.write(msg + "\n")

                    # Guardar el código fuente y la información del crash
                    with open(f"kernelhunter_crashes/gen{gen_id:04d}_prog{i:04d}.c", "w") as fc:
                        fc.write(stub)

                    # Guardar información detallada del crash
                    save_crash_info(gen_id, i, shellcode, crash_type, stderr_text,
                                   stdout_text, result.returncode, is_system_impact, parent)

                    crashes += 1
                    if USE_RL_WEIGHTS:
                        crash_info = {
                            "crash_type": crash_type,
                            "system_impact": is_system_impact
                        }
                        reward = calculate_reward(crash_info)
                        attack_success[last_attack_type] += reward
                        mutation_success[last_mutation_type] += reward
                        sliding_attack_reward.add_result(last_attack_type, reward)
                        sliding_mutation_reward.add_result(last_mutation_type, reward)
                        update_q_value(attack_q_values, attack_counts, attack_index[last_attack_type], reward)
                        update_q_value(mutation_q_values, mutation_counts, mutation_index[last_mutation_type], reward)
                        update_rl_weights()
                if USE_RL_WEIGHTS and result.returncode == 0:
                    sliding_attack_reward.add_result(last_attack_type, 0)
                    sliding_mutation_reward.add_result(last_mutation_type, 0)
                    update_q_value(attack_q_values, attack_counts, attack_index[last_attack_type], 0)
                    update_q_value(mutation_q_values, mutation_counts, mutation_index[last_mutation_type], 0)

            # Manejar otros tipos de errores
            except subprocess.TimeoutExpired:
                crash_type = "TIMEOUT"
                shellcode_hex = print_shellcode_hex(shellcode, escape_format=True)
                msg = f"Crash {i:04d}: TIMEOUT | Shellcode: {shellcode_hex}"
                print(color_text(msg, "red"))
                crash_log.write(msg + "\n")
                with open(f"kernelhunter_crashes/gen{gen_id:04d}_prog{i:04d}.c", "w") as fc:
                    fc.write(stub)
                crashes += 1
                crash_types_counter[crash_type] += 1
                if USE_RL_WEIGHTS:
                    crash_info = {"crash_type": crash_type, "system_impact": False}
                    reward = calculate_reward(crash_info)
                    attack_success[last_attack_type] += reward
                    mutation_success[last_mutation_type] += reward
                    sliding_attack_reward.add_result(last_attack_type, reward)
                    sliding_mutation_reward.add_result(last_mutation_type, reward)
                    update_q_value(attack_q_values, attack_counts, attack_index[last_attack_type], reward)
                    update_q_value(mutation_q_values, mutation_counts, mutation_index[last_mutation_type], reward)
                    update_rl_weights()

            except subprocess.CalledProcessError:
                crash_type = "COMPILE_ERROR"
                shellcode_hex = print_shellcode_hex(shellcode, escape_format=True)
                msg = f"Crash {i:04d}: COMPILATION ERROR | Shellcode: {shellcode_hex}"
                print(color_text(msg, "red"))
                crash_log.write(msg + "\n")
                crashes += 1
                crash_types_counter[crash_type] += 1
                if USE_RL_WEIGHTS:
                    crash_info = {"crash_type": crash_type, "system_impact": False}
                    reward = calculate_reward(crash_info)
                    attack_success[last_attack_type] += reward
                    mutation_success[last_mutation_type] += reward
                    sliding_attack_reward.add_result(last_attack_type, reward)
                    sliding_mutation_reward.add_result(last_mutation_type, reward)
                    update_q_value(attack_q_values, attack_counts, attack_index[last_attack_type], reward)
                    update_q_value(mutation_q_values, mutation_counts, mutation_index[last_mutation_type], reward)
                    update_rl_weights()

    # Asegurar que siempre hay población
    if not new_population:
        # Si todos crashearon, reiniciar con la población base pero con algunos modificados
        print(color_text(f"[GEN {gen_id}] Todos crashearon. Reiniciando población.", "yellow"))
        new_population = [BASE_SHELLCODE + EXIT_SYSCALL]
        for _ in range(min(5, len(base_population))):
            parent = choice(base_population)
            new_population.append(mutate_shellcode(parent))

    # Calcular y mostrar métricas
    crash_rate = crashes / NUM_PROGRAMS
    avg_length = sum(shellcode_lengths) / len(shellcode_lengths) if shellcode_lengths else 0



    print(color_text(f"[GEN {gen_id}] Crash rate: {crash_rate*100:.1f}% | Sys impacts: {system_impacts} | Avg length: {avg_length:.1f}", "cyan"))
    print(color_text(f"[GEN {gen_id}] Crash types: {dict(crash_types_counter.most_common(3))}", "cyan"))

    rl_data = {
        "attack_q_values": list(attack_q_values),
        "mutation_q_values": list(mutation_q_values),
        "epsilon": get_epsilon(gen_id)
    } if USE_RL_WEIGHTS else {
        "attack_weights": attack_weights,
        "mutation_weights": mutation_weights,
        "epsilon": get_epsilon(gen_id)
    }

    diversity_data = genetic_reservoir.get_diversity_stats()

    print(f"[DEBUG] Llamando a log_generation con gen_id={gen_id}")
    print(f"[DEBUG] population_size={NUM_PROGRAMS} crash_rate={crash_rate}")
    print(f"[DEBUG] base_population={len(base_population)} new_population={len(new_population)}")
    print(f"[DEBUG] crash_types={dict(crash_types_counter)}")
    print(f"[DEBUG] attack_stats={dict(generation_attack_counter)}")
    print(f"[DEBUG] mutation_stats={dict(generation_mutation_counter)}")
    print(f"[DEBUG] rl_data={rl_data}")
    print(f"[DEBUG] diversity_data={diversity_data}")

    pythonlogger.log_generation(
        generation_id=gen_id,
        population_size=NUM_PROGRAMS,
        crash_rate=crash_rate,
        system_impacts=system_impacts,
        avg_shellcode_length=avg_length,
        crash_types=dict(crash_types_counter),
        attack_stats=dict(generation_attack_counter),
        mutation_stats=dict(generation_mutation_counter),
        rl_weights=rl_data,
        diversity_metrics=diversity_data
    )

    # Actualizar métricas
    metrics["generations"].append(gen_id)
    metrics["crash_rates"].append(crash_rate)
    metrics["system_impacts"].append(system_impacts)
    metrics["shellcode_lengths"].append(avg_length)
    metrics["crash_types"][gen_id] = dict(crash_types_counter)
    metrics.setdefault("attack_stats", {})[gen_id] = dict(generation_attack_counter)
    metrics.setdefault("mutation_stats", {})[gen_id] = dict(generation_mutation_counter)
    for choice_type in total_attack_counter:
        metrics.setdefault("attack_totals", {})[choice_type] = total_attack_counter[choice_type]
    for mutation_type in total_mutation_counter:
        metrics.setdefault("mutation_totals", {})[mutation_type] = total_mutation_counter[mutation_type]
    write_metrics()

    update_rl_weights()

    # Guardar checkpoint periódico
    if gen_id % CHECKPOINT_INTERVAL == 0:
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)

    # When diversity is needed, get samples from the reservoir
    if crash_rate < 0.05 and len(genetic_reservoir) >= 5:
        reservoir_samples = genetic_reservoir.get_diverse_sample(n=5)
        # Add these to your population or use them for new mutations
        for sample in reservoir_samples:
            new_population.append(mutate_shellcode(sample))

    # Save state periodically
    if gen_id % 10 == 0:
        genetic_reservoir.save_to_file(get_reservoir_file())

    # Verificar y reiniciar individuos estancados
    if crash_rate == 0 and gen_id > 0:
        # Identificar individuos estancados
        stagnant_individuals = []
        non_stagnant = []

        # Actualizar contadores y clasificar individuos
        for shellcode in base_population:
            shellcode_id = shellcode.hex()[:20]  # Usar los primeros bytes como identificador

            if shellcode_id not in individual_zero_crash_counts:
                individual_zero_crash_counts[shellcode_id] = 0

            individual_zero_crash_counts[shellcode_id] += 1

            # Verificar si este individuo está estancado
            if individual_zero_crash_counts[shellcode_id] >= MAX_INDIVIDUAL_ZERO_CRASH_GENERATIONS:
                stagnant_individuals.append(shellcode)
            else:
                non_stagnant.append(shellcode)

        # Si hay individuos estancados, reemplazarlos
        if len(stagnant_individuals) > 0:
            print(color_text(f"[GEN {gen_id}] ¡ALERTA! {len(stagnant_individuals)} individuos estancados identificados.", "yellow"))
            print(color_text(f"[GEN {gen_id}] Manteniendo {len(non_stagnant)} individuos, creando {len(stagnant_individuals)} nuevos.", "yellow"))

            # Iniciar con los individuos no estancados
            new_population = non_stagnant.copy()

            # Generar nuevos individuos aplicando mutaciones a la población existente
            for _ in range(len(stagnant_individuals)):
                parent = choice(base_population)
                mutated = mutate_shellcode(parent)
                new_population.append(mutated)

            # Limpiar contadores antiguos para ahorrar memoria
            if len(individual_zero_crash_counts) > 1000:  # Límite arbitrario
                individual_zero_crash_counts = {k: v for k, v in individual_zero_crash_counts.items()
                                             if v >= MAX_INDIVIDUAL_ZERO_CRASH_GENERATIONS // 2}
    elif crash_rate > 0:
        # Hubo crashes, reiniciar contadores para todos los individuos
        for shellcode in base_population:
            shellcode_id = shellcode.hex()[:20]
            if shellcode_id in individual_zero_crash_counts:
                individual_zero_crash_counts[shellcode_id] = 0

    # Si no hay población (todos crashearon), reiniciar con población base
    if not new_population:
        print(color_text(f"[GEN {gen_id}] Todos crashearon. Reiniciando población.", "yellow"))
        new_population = [BASE_SHELLCODE + EXIT_SYSCALL]
        for _ in range(min(5, len(base_population))):
            parent = choice(base_population)
            new_population.append(mutate_shellcode(parent))

    # Attempt to log generation statistics
    try:
        pythonlogger.log_generation(
            generation_id=gen_id,
            population_size=len(new_population),
            crash_rate=crash_rate,
            system_impacts=system_impacts,
            avg_shellcode_length=avg_length,
            crash_types=dict(crash_types_counter),
            attack_stats=dict(generation_attack_counter),
            mutation_stats=dict(generation_mutation_counter)
        )
        print(f"[DEBUG] \u2713 Logger llamado para generación {gen_id}")
    except Exception as e:
        print(f"[DEBUG] \u2717 Error en logger gen {gen_id}: {e}")

    combined_population = deduplicate_population(base_population + new_population)
    if len(combined_population) > MAX_POPULATION_SIZE:
        combined_population = random.sample(combined_population, MAX_POPULATION_SIZE)
    return combined_population

async def run_generation_parallel(gen_id, base_population):
    """Versión paralela de run_generation usando asyncio y ProcessPoolExecutor"""
    global individual_zero_crash_counts, generation_attack_counter, generation_mutation_counter
    global attack_success, mutation_success, current_generation
    current_generation = gen_id

    generation_attack_counter = Counter()
    generation_mutation_counter = Counter()
    attack_success.clear()
    mutation_success.clear()

    gen_path = os.path.join(OUTPUT_DIR, f"gen_{gen_id:04d}")
    os.makedirs(gen_path, exist_ok=True)

    new_population = []
    crashes = 0
    crash_types_counter = Counter()
    system_impacts = 0
    shellcode_lengths = []

    with open(LOG_FILE, "a") as log, open(CRASH_LOG, "a") as crash_log:
        log.write(f"\n[GEN {gen_id}] {'='*50}\n")
        crash_log.write(f"\n[GEN {gen_id}] {'='*50}\n")

        max_workers = min(multiprocessing.cpu_count(), NUM_PROGRAMS)
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = []
            for i in range(NUM_PROGRAMS):
                if i % 5 == 0:
                    print(f"Generando individuos {i}/{NUM_PROGRAMS}", end="\r")

                # Selección natural pura - sin sesgos artificiales
                parent = choice(base_population)

                tasks.append(process_single_program_async(executor, i, parent, gen_id, gen_path))

            try:
                results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=30)
            except asyncio.TimeoutError:
                print(color_text(f"[GEN {gen_id}] Generación superó tiempo límite", "red"))
                for t in tasks:
                    t.cancel()
                results = []
                for t in tasks:
                    try:
                        results.append(await t)
                    except Exception:
                        pass

        for res in results:
            shellcode = res["shellcode"]
            shellcode_lengths.append(len(shellcode))

            if res["survived"]:
                new_population.append(shellcode)
                shellcode_hex = print_shellcode_hex(shellcode, escape_format=True)
                log.write(f"Survivor {res['program_id']:04d}: {shellcode_hex} (len: {len(shellcode)})\n")
                if res["program_id"] < 2:
                    print(color_text(f"Survivor sample {res['program_id']:04d}: {shellcode_hex}", "green"))
                if USE_RL_WEIGHTS:
                    update_q_value(attack_q_values, attack_counts, attack_index[last_attack_type], 0)
                    update_q_value(mutation_q_values, mutation_counts, mutation_index[last_mutation_type], 0)
                    update_rl_weights()
                continue

            crash_info = res.get("crash_info") or {}
            crash_type = crash_info.get("crash_type", "UNKNOWN")
            if crash_type not in ("TIMEOUT", "COMPILE_ERROR") and res["result"] != "TIMEOUT" and res["result"] != "COMPILE_ERROR":
                stderr_text = crash_info.get("stderr", "")
                stdout_text = crash_info.get("stdout", "")
                is_system_impact = crash_info.get("system_impact", False)
                if is_system_impact:
                    system_impacts += 1
                    genetic_reservoir.add(crash_info.get("parent"), crash_info)
                msg = f"Crash {res['program_id']:04d}: {crash_type} | Shellcode: {print_shellcode_hex(shellcode, escape_format=True)}"
                if is_system_impact:
                    msg = f"[SYSTEM IMPACT] {msg}"
                crash_types_counter[crash_type] += 1
                if is_system_impact:
                    print(color_text(msg, "magenta"))
                else:
                    print(color_text(msg, "red"))
                crash_log.write(msg + "\n")
                with open(f"kernelhunter_crashes/gen{gen_id:04d}_prog{res['program_id']:04d}.c", "w") as fc:
                    fc.write(res["stub"])
                save_crash_info(gen_id, res['program_id'], shellcode, crash_type, stderr_text, stdout_text, crash_info.get("return_code"), is_system_impact, crash_info.get("parent"))
                crashes += 1
                if USE_RL_WEIGHTS:
                    reward = calculate_reward(crash_info)
                    attack_success[last_attack_type] += reward
                    mutation_success[last_mutation_type] += reward
                    sliding_attack_reward.add_result(last_attack_type, reward)
                    sliding_mutation_reward.add_result(last_mutation_type, reward)
                    update_q_value(attack_q_values, attack_counts, attack_index[last_attack_type], reward)
                    update_q_value(mutation_q_values, mutation_counts, mutation_index[last_mutation_type], reward)
                    update_rl_weights()
            else:
                crash_types_counter[crash_type] += 1
                msg = f"Crash {res['program_id']:04d}: {crash_type} | Shellcode: {print_shellcode_hex(shellcode, escape_format=True)}"
                print(color_text(msg, "red"))
                crash_log.write(msg + "\n")
                with open(f"kernelhunter_crashes/gen{gen_id:04d}_prog{res['program_id']:04d}.c", "w") as fc:
                    fc.write(res["stub"])
                crashes += 1
                if USE_RL_WEIGHTS:
                    reward = calculate_reward({"crash_type": crash_type, "system_impact": False})
                    attack_success[last_attack_type] += reward
                    mutation_success[last_mutation_type] += reward
                    sliding_attack_reward.add_result(last_attack_type, reward)
                    sliding_mutation_reward.add_result(last_mutation_type, reward)
                    update_q_value(attack_q_values, attack_counts, attack_index[last_attack_type], reward)
                    update_q_value(mutation_q_values, mutation_counts, mutation_index[last_mutation_type], reward)
                    update_rl_weights()

    if not new_population:
        print(color_text(f"[GEN {gen_id}] Todos crashearon. Reiniciando población.", "yellow"))
        new_population = [BASE_SHELLCODE + EXIT_SYSCALL]
        for _ in range(min(5, len(base_population))):
            parent = choice(base_population)
            new_population.append(mutate_shellcode(parent))

    crash_rate = crashes / NUM_PROGRAMS
    avg_length = sum(shellcode_lengths) / len(shellcode_lengths) if shellcode_lengths else 0

    print(color_text(f"[GEN {gen_id}] Crash rate: {crash_rate*100:.1f}% | Sys impacts: {system_impacts} | Avg length: {avg_length:.1f}", "cyan"))
    print(color_text(f"[GEN {gen_id}] Crash types: {dict(crash_types_counter.most_common(3))}", "cyan"))

    metrics["generations"].append(gen_id)
    metrics["crash_rates"].append(crash_rate)
    metrics["system_impacts"].append(system_impacts)
    metrics["shellcode_lengths"].append(avg_length)
    metrics["crash_types"][gen_id] = dict(crash_types_counter)
    metrics.setdefault("attack_stats", {})[gen_id] = dict(generation_attack_counter)
    metrics.setdefault("mutation_stats", {})[gen_id] = dict(generation_mutation_counter)
    for choice_type in total_attack_counter:
        metrics.setdefault("attack_totals", {})[choice_type] = total_attack_counter[choice_type]
    for mutation_type in total_mutation_counter:
        metrics.setdefault("mutation_totals", {})[mutation_type] = total_mutation_counter[mutation_type]
    write_metrics()

    update_rl_weights()

    if gen_id % CHECKPOINT_INTERVAL == 0:
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)

    if crash_rate < 0.05 and len(genetic_reservoir) >= 5:
        reservoir_samples = genetic_reservoir.get_diverse_sample(n=5)
        for sample in reservoir_samples:
            new_population.append(mutate_shellcode(sample))

    if gen_id % 10 == 0:
        genetic_reservoir.save_to_file(get_reservoir_file())

    if crash_rate == 0 and gen_id > 0:
        stagnant_individuals = []
        non_stagnant = []
        for shellcode in base_population:
            shellcode_id = shellcode.hex()[:20]
            if shellcode_id not in individual_zero_crash_counts:
                individual_zero_crash_counts[shellcode_id] = 0
            individual_zero_crash_counts[shellcode_id] += 1
            if individual_zero_crash_counts[shellcode_id] >= MAX_INDIVIDUAL_ZERO_CRASH_GENERATIONS:
                stagnant_individuals.append(shellcode)
            else:
                non_stagnant.append(shellcode)
        if len(stagnant_individuals) > 0:
            print(color_text(f"[GEN {gen_id}] ¡ALERTA! {len(stagnant_individuals)} individuos estancados identificados.", "yellow"))
            print(color_text(f"[GEN {gen_id}] Manteniendo {len(non_stagnant)} individuos, creando {len(stagnant_individuals)} nuevos.", "yellow"))
            new_population = non_stagnant.copy()
            for _ in range(len(stagnant_individuals)):
                parent = choice(base_population)
                mutated = mutate_shellcode(parent)
                new_population.append(mutated)
            if len(individual_zero_crash_counts) > 1000:
                individual_zero_crash_counts = {k: v for k, v in individual_zero_crash_counts.items() if v >= MAX_INDIVIDUAL_ZERO_CRASH_GENERATIONS // 2}
    elif crash_rate > 0:
        for shellcode in base_population:
            shellcode_id = shellcode.hex()[:20]
            if shellcode_id in individual_zero_crash_counts:
                individual_zero_crash_counts[shellcode_id] = 0

    if not new_population:
        print(color_text(f"[GEN {gen_id}] Todos crashearon. Reiniciando población.", "yellow"))
        new_population = [BASE_SHELLCODE + EXIT_SYSCALL]
        for _ in range(min(5, len(base_population))):
            parent = choice(base_population)
            new_population.append(mutate_shellcode(parent))

    # Attempt to log generation statistics
    try:
        pythonlogger.log_generation(
            generation_id=gen_id,
            population_size=len(new_population),
            crash_rate=crash_rate,
            system_impacts=system_impacts,
            avg_shellcode_length=avg_length,
            crash_types=dict(crash_types_counter),
            attack_stats=dict(generation_attack_counter),
            mutation_stats=dict(generation_mutation_counter)
        )
        print(f"[DEBUG] \u2713 Logger llamado para generación {gen_id}")
    except Exception as e:
        print(f"[DEBUG] \u2717 Error en logger gen {gen_id}: {e}")

    combined_population = deduplicate_population(base_population + new_population)
    if len(combined_population) > MAX_POPULATION_SIZE:
        combined_population = random.sample(combined_population, MAX_POPULATION_SIZE)
    return combined_population

def analyze_crash_files():
    """Analiza los archivos de crash guardados para buscar patrones"""
    crash_dir = "kernelhunter_critical"
    crash_files = [f for f in os.listdir(crash_dir) if f.endswith(".json")]

    if not crash_files:
        print(color_text("No se encontraron crashes críticos para analizar.", "yellow"))
        return

    print(color_text(f"\nAnalizando {len(crash_files)} crashes con potencial impacto a nivel de sistema...", "cyan"))

    # Análisis básico de patrones
    crash_signals = Counter()
    shellcode_segments = Counter()
    avg_length = 0

    for crash_file in crash_files:
        with open(os.path.join(crash_dir, crash_file), 'r') as f:
            data = json.load(f)

            # Contabilizar tipos de crash
            crash_signals[data.get('crash_type', 'UNKNOWN')] += 1

            # Analizar shellcode
            shellcode_hex = data.get('shellcode_hex', '')
            avg_length += len(shellcode_hex) // 2

            # Buscar patrones comunes (segmentos de 6 bytes)
            if len(shellcode_hex) >= 12:  # 6 bytes = 12 caracteres hex
                for i in range(0, len(shellcode_hex) - 12, 2):
                    segment = shellcode_hex[i:i+12]
                    shellcode_segments[segment] += 1

    avg_length = avg_length / len(crash_files) if crash_files else 0

    print(color_text("\nResumen de análisis:", "cyan"))
    print(color_text(f"- Promedio de longitud de shellcode: {avg_length:.1f} bytes", "cyan"))
    print(color_text(f"- Señales más comunes: {dict(crash_signals.most_common(3))}", "cyan"))

    if shellcode_segments:
        print(color_text(f"- Segmentos de shellcode más comunes:", "cyan"))
        for segment, count in shellcode_segments.most_common(5):
            if count > 1:
                # Convertir a formato \x01\x02
                hex_bytes = bytes.fromhex(segment)
                escaped_format = ''.join(f'\\x{b:02x}' for b in hex_bytes)
                print(color_text(f"  {escaped_format}: {count} apariciones", "cyan"))

        # Intentar decodificar los segmentos más comunes
        print(color_text("\n- Interpretación de segmentos comunes:", "cyan"))
        for segment, count in shellcode_segments.most_common(3):
            if count > 1:
                try:
                    # Convertir hex a bytes
                    segment_bytes = bytes.fromhex(segment)
                    # Intentar desensamblar
                    interpretation = interpret_instruction(segment_bytes)
                    escaped_format = ''.join(f'\\x{b:02x}' for b in segment_bytes)
                    print(color_text(f"  {escaped_format}: Posiblemente '{interpretation}'", "cyan"))
                except Exception as e:
                    pass

async def main_async():
    banner = """
  _  __                    _   _   _             _
 | |/ /___ _ __ _ __   ___| | | | | |_   _ _ __ | |_ ___ _ __
 | ' // _ \ '__| '_ \ / _ \ | | |_| | | | | '_ \| __/ _ \ '__|
 | . \  __/ |  | | | |  __/ | |  _  | |_| | | | | ||  __/ |
 |_|\_\___|_|  |_| |_|\___|_| |_| |_|\__,_|_| |_|\__\___|_|

 Fuzzer evolutivo para vulnerabilidades del sistema operativo
 -------------------------------------------------------------
"""
    print(banner)
    global pythonlogger
    pythonlogger = PerformanceLogger(f"kernelhunter_{'rl' if USE_RL_WEIGHTS else 'normal'}_{int(time.time())}")
    pythonlogger.set_rl_mode(USE_RL_WEIGHTS)
    print(f"Configuración: {NUM_PROGRAMS} programas por generación, timeout: {TIMEOUT}s")

    initial = BASE_SHELLCODE
    population = [initial]

    with open(LOG_FILE, "w") as log:
        log.write("KernelHunter - Log de shellcodes sobrevivientes\n=======================================\n")
    with open(CRASH_LOG, "w") as crash_log:
        crash_log.write("KernelHunter - Log de crashes clasificados\n=======================================\n")

    # Record initial weight configuration
    update_rl_weights()
    # Add reservoir samples to initial population
    if len(genetic_reservoir) > 0:
        # Get a diverse sample from the reservoir
        samples = genetic_reservoir.get_diverse_sample(n=min(10, len(genetic_reservoir)))
        
        # Add these samples to the initial population
        population.extend(samples)
        print(color_text(f"Added {len(samples)} diverse individuals from reservoir to initial population", "green"))

    try:
        for gen in range(MAX_GENERATIONS):
            print(color_text(f"\nGeneración {gen}/{MAX_GENERATIONS} (población: {len(population)})", "cyan"))
            population = await run_generation_parallel(gen, population)

    except KeyboardInterrupt:
        print(color_text("\n\nProceso interrumpido por el usuario.", "red"))

    finally:
        print(color_text("\nGuardando métricas finales...", "cyan"))
        write_metrics()

        print(color_text("\nEvolución completada.", "green"))
        analyze_crash_files()

        print(color_text("\nResumen final:", "cyan"))
        print(color_text(f"- Total de generaciones: {len(metrics['generations'])}", "cyan"))
        if metrics['system_impacts']:
            total_impacts = sum(metrics['system_impacts'])
            print(color_text(f"- Total de impactos potenciales al sistema: {total_impacts}", "magenta"))
            print(color_text(f"- Directorio de crashes críticos: kernelhunter_critical/", "magenta"))

        pythonlogger.finalize_session()

        print(color_text("\nKernelHunter ha completado su ejecución.", "green"))

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KernelHunter evolutionary fuzzer")
    parser.add_argument("--use-rl-weights", action="store_true",
                        help="enable reinforcement learning weight adjustments")
    args = parser.parse_args()

    if args.use_rl_weights:
        USE_RL_WEIGHTS = True
        attack_weights = DEFAULT_ATTACK_WEIGHTS.copy()
        mutation_weights = DEFAULT_MUTATION_WEIGHTS.copy()
        attack_q_values, mutation_q_values = initialize_bandit_values()
        attack_counts = [0] * len(ATTACK_OPTIONS)
        mutation_counts = [0] * len(MUTATION_TYPES)

    main()
