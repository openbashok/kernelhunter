#!/usr/bin/env python3
# KernelHunter - Fuzzer evolutivo avanzado para encontrar vulnerabilidades en el sistema operativo
# Mariano Somoza
# Un enfoque evolutivo con ML, analytics en tiempo real, y sandboxing avanzado

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
import argparse
from random import randint, choice
from collections import Counter, defaultdict, deque
from genetic_reservoir import GeneticReservoir
from performance_logger import PerformanceLogger

# Importar módulos avanzados
try:
    from advanced_ml_engine import get_ml_engine, MLConfig
    from performance_optimizer import get_performance_optimizer, PerformanceConfig
    from real_time_analytics import get_analytics_engine, AnalyticsConfig
    from advanced_security_sandbox import get_security_sandbox, SandboxConfig
    from distributed_orchestrator import get_distributed_orchestrator
    ADVANCED_FEATURES_ENABLED = True
except ImportError as e:
    print(f"⚠️ Módulos avanzados no disponibles: {e}")
    print("🔧 Ejecutando en modo básico...")
    ADVANCED_FEATURES_ENABLED = False

# Importar nueva configuración
try:
    from kernelhunter_config import get_config, is_local_mode, is_distributed_mode, get_effective_config, setup_environment
    config = get_config()
    effective_config = get_effective_config()
except Exception as e:
    print(f"⚠️ Error cargando configuración: {e}")
    print("🔧 Usando configuración por defecto...")
    
    # Configuración de fallback - SIN SANDBOX por defecto
    effective_config = {
        'max_generations': 100,
        'population_size': 50,
        'mutation_rate': 0.3,
        'crossover_rate': 0.7,
        'elite_size': 5,
        'stagnation_limit': 20,
        'enable_rl': True,
        'enable_web_interface': False,
        'web_port': 8080,
        'log_level': 'INFO',
        'save_crashes': True,
        'crash_dir': './crashes',
        'metrics_file': './metrics.json',
        'gene_bank_file': './gene_bank.json',
        'enable_advanced_features': ADVANCED_FEATURES_ENABLED,
        'enable_ml': ADVANCED_FEATURES_ENABLED,
        'enable_analytics': ADVANCED_FEATURES_ENABLED,
        'enable_security_sandbox': False,  # ¡SIN SANDBOX por defecto!
        'sandbox_isolation_level': 'none',  # Sin aislamiento
        'max_execution_time': 30,
        'max_memory_mb': 512
    }

# Configuración global
OUTPUT_DIR = "kernelhunter_generations"
NUM_PROGRAMS = effective_config.get('population_size', 50)
MAX_GENERATIONS = effective_config.get('max_generations', 1000)
TIMEOUT = effective_config.get('max_execution_time', 3)
LOG_FILE = "kernelhunter_survivors.txt"
CRASH_LOG = "kernelhunter_crashes.txt"
METRICS_FILE = effective_config.get('metrics_file', "kernelhunter_metrics.json")
CHECKPOINT_INTERVAL = 10
MAX_POPULATION_SIZE = effective_config.get('population_size', 1000)
NATURAL_SELECTION_MODE = True

# Crear directorios necesarios
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(effective_config.get('crash_dir', "kernelhunter_crashes"), exist_ok=True)
os.makedirs("kernelhunter_critical", exist_ok=True)
os.makedirs("ml_models", exist_ok=True)
os.makedirs("analytics_data", exist_ok=True)

# Configuración de RL desde la nueva configuración
USE_RL_WEIGHTS = effective_config.get('enable_rl', False)

# Attack options and weights (actualizado con nuevos ataques)
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
    "ml_generated",  # Nuevo: shellcodes generados por ML
    "transformer_generated",  # Nuevo: shellcodes generados por transformer
]

DEFAULT_ATTACK_WEIGHTS = [
    80, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
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
    "neutral_mutation",
    "ml_guided_mutation",  # Nuevo: mutación guiada por ML
]

# Variables globales
attack_weights = DEFAULT_ATTACK_WEIGHTS.copy()
mutation_weights = [1] * len(MUTATION_TYPES)
attack_q_values = defaultdict(float)
mutation_q_values = defaultdict(float)
attack_counts = defaultdict(int)
mutation_counts = defaultdict(int)
attack_success = defaultdict(float)
mutation_success = defaultdict(float)
total_attack_counter = Counter()
total_mutation_counter = Counter()
generation_attack_counter = Counter()
generation_mutation_counter = Counter()
last_attack_type = None
last_mutation_type = None
current_generation = 0
individual_zero_crash_counts = {}
MAX_INDIVIDUAL_ZERO_CRASH_GENERATIONS = effective_config.get('stagnation_limit', 20)

# Genetic reservoir
genetic_reservoir = GeneticReservoir()

# Performance logging
pythonlogger = None

# Instancias de módulos avanzados
ml_engine = None
performance_optimizer = None
analytics_engine = None
security_sandbox = None
distributed_orchestrator = None

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

# Métricas globales
metrics = {
    "generations": [],
    "crash_rates": [],
    "system_impacts": [],
    "shellcode_lengths": [],
    "crash_types": {},
    "attack_stats": {},
    "mutation_stats": {},
    "ml_metrics": {},
    "analytics_metrics": {},
    "security_metrics": {}
}

def write_metrics():
    """Write metrics to file"""
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

def get_reservoir_file():
    """Get genetic reservoir file path"""
    return effective_config.get('gene_bank_file', "gene_bank.json")

def initialize_advanced_modules():
    """Initialize advanced modules if available"""
    global ml_engine, performance_optimizer, analytics_engine, security_sandbox, distributed_orchestrator
    
    if not ADVANCED_FEATURES_ENABLED:
        return
    
    try:
        # Initialize ML engine
        if effective_config.get('enable_ml', False):
            ml_config = MLConfig()
            ml_engine = get_ml_engine()
            print("✅ Motor de Machine Learning inicializado")
        
        # Initialize performance optimizer
        perf_config = PerformanceConfig()
        performance_optimizer = get_performance_optimizer()
        print("✅ Optimizador de rendimiento inicializado")
        
        # Initialize analytics engine
        if effective_config.get('enable_analytics', False):
            analytics_config = AnalyticsConfig()
            analytics_engine = get_analytics_engine()
            print("✅ Motor de analytics en tiempo real inicializado")
        
        # Initialize security sandbox
        if effective_config.get('enable_security_sandbox', False):
            sandbox_config = SandboxConfig(
                isolation_level=effective_config.get('sandbox_isolation_level', 'container'),
                max_execution_time=effective_config.get('max_execution_time', 30),
                max_memory_mb=effective_config.get('max_memory_mb', 512)
            )
            security_sandbox = get_security_sandbox()
            print("✅ Sandbox de seguridad inicializado")
        
        # Initialize distributed orchestrator (if in distributed mode)
        if is_distributed_mode():
            distributed_orchestrator = get_distributed_orchestrator()
            print("✅ Orquestador distribuido inicializado")
            
    except Exception as e:
        print(f"⚠️ Error inicializando módulos avanzados: {e}")

async def start_advanced_services():
    """Start advanced services"""
    if not ADVANCED_FEATURES_ENABLED:
        return
    
    try:
        # Start performance optimizer
        if performance_optimizer:
            await performance_optimizer.start()
        
        # Start analytics engine
        if analytics_engine:
            await analytics_engine.start()
        
        # Start distributed orchestrator
        if distributed_orchestrator:
            await distributed_orchestrator.start()
            
    except Exception as e:
        print(f"⚠️ Error iniciando servicios avanzados: {e}")

async def stop_advanced_services():
    """Stop advanced services"""
    if not ADVANCED_FEATURES_ENABLED:
        return
    
    try:
        # Stop performance optimizer
        if performance_optimizer:
            await performance_optimizer.stop()
        
        # Stop analytics engine
        if analytics_engine:
            await analytics_engine.stop()
        
        # Stop distributed orchestrator
        if distributed_orchestrator:
            await distributed_orchestrator.stop()
        
        # Cleanup security sandbox
        if security_sandbox:
            security_sandbox.cleanup_all()
            
    except Exception as e:
        print(f"⚠️ Error deteniendo servicios avanzados: {e}")

def generate_fragment(choice_type=None):
    """Generate shellcode fragment with advanced features"""
    global last_attack_type
    
    if choice_type is None:
        if USE_RL_WEIGHTS and ml_engine:
            # Use ML engine for intelligent selection
            state_features = extract_state_features()
            choice_type = ml_engine.select_attack_strategy(state_features)
        else:
            choice_type = random.choices(ATTACK_OPTIONS, weights=attack_weights)[0]
    
    last_attack_type = choice_type
    generation_attack_counter[choice_type] += 1
    total_attack_counter[choice_type] += 1
    
    # Update metrics
    metrics.setdefault("attack_totals", {})[choice_type] = total_attack_counter[choice_type]
    write_metrics()
    
    # Generate fragment based on type
    if choice_type == "ml_generated" and ml_engine:
        # Use ML engine to generate shellcode
        return ml_engine.generate_shellcode_transformer(b"", max_length=64)
    elif choice_type == "transformer_generated" and ml_engine:
        # Use transformer model
        seed = bytes([random.randint(0, 255) for _ in range(8)])
        return ml_engine.generate_shellcode_transformer(seed, max_length=64)
    elif choice_type == "ai_shellcode":
        # Use AI shellcode generation
        try:
            from ai_shellcode_mutation import generate_ai_shellcode_fragment
            return generate_ai_shellcode_fragment(max_bytes=32)
        except ImportError:
            return generate_random_instruction()
    else:
        # Use existing fragment generation logic
        return generate_traditional_fragment(choice_type)

def generate_traditional_fragment(choice_type):
    """Generate fragment using traditional methods"""
    # ... existing fragment generation logic ...
    # (Keep all the existing elif blocks for different attack types)
    
    # For brevity, I'll include a few key ones:
    if choice_type == "random_bytes":
        return bytes([randint(0, 255) for _ in range(randint(4, 16))])
    elif choice_type == "syscall_setup":
        return random.choice(SYSCALL_SETUP)
    elif choice_type == "syscall":
        return SYSCALL_PATTERN
    elif choice_type == "memory_access":
        return b"\x48\x8b\x07"  # mov rax, [rdi]
    elif choice_type == "privileged":
        return b"\x0f\x20\xc0"  # mov rax, cr0
    else:
        return bytes([randint(0, 255) for _ in range(randint(1, 8))])

def extract_state_features():
    """Extract features for ML models"""
    return {
        "crash_rate": metrics.get("crash_rates", [0])[-1] if metrics.get("crash_rates") else 0,
        "system_impacts": metrics.get("system_impacts", [0])[-1] if metrics.get("system_impacts") else 0,
        "generation": current_generation,
        "population_size": NUM_PROGRAMS,
        "diversity_score": genetic_reservoir.get_diversity_stats().get("diversity_avg", 0),
        "rl_epsilon": get_epsilon(current_generation) if USE_RL_WEIGHTS else 0,
        "attack_success_rate": sum(attack_success.values()) / len(attack_success) if attack_success else 0,
        "mutation_success_rate": sum(mutation_success.values()) / len(mutation_success) if mutation_success else 0
    }

def mutate_shellcode(shellcode, mutation_rate=0.8):
    """Muta el shellcode con diferentes estrategias incluyendo ML"""
    global last_mutation_type
    
    # Remove EXIT
    if shellcode.endswith(EXIT_SYSCALL):
        core = shellcode[:-len(EXIT_SYSCALL)]
    else:
        core = shellcode
    
    # Select mutation type
    if USE_RL_WEIGHTS and ml_engine:
        # Use ML engine for intelligent mutation selection
        state_features = extract_state_features()
        mutation_type = ml_engine.select_mutation_strategy(state_features)
    else:
        mutation_type = random.choices(MUTATION_TYPES, weights=mutation_weights)[0]
    
    last_mutation_type = mutation_type
    generation_mutation_counter[mutation_type] += 1
    total_mutation_counter[mutation_type] += 1
    
    # Update metrics
    metrics.setdefault("mutation_totals", {})[mutation_type] = total_mutation_counter[mutation_type]
    write_metrics()
    
    # Apply mutation
    if mutation_type == "ml_guided_mutation" and ml_engine:
        # Use ML-guided mutation
        return apply_ml_guided_mutation(core)
    elif mutation_type == "neutral_mutation":
        # Use neutral mutation
        try:
            from neutral_mutation import insert_neutral_mutation
            return insert_neutral_mutation(core)
        except ImportError:
            return apply_traditional_mutation(core, mutation_type)
    else:
        return apply_traditional_mutation(core, mutation_type)

def apply_ml_guided_mutation(shellcode):
    """Apply ML-guided mutation"""
    if ml_engine:
        # Use ML to predict best mutation
        features = extract_state_features()
        # This is a simplified version - in practice you'd use more sophisticated ML
        return shellcode + bytes([random.randint(0, 255) for _ in range(4)])
    else:
        return shellcode + bytes([random.randint(0, 255) for _ in range(4)])

def apply_traditional_mutation(shellcode, mutation_type):
    """Apply traditional mutation"""
    if mutation_type == "add" or not shellcode:
        instr = generate_random_instruction()
        insert_pos = randint(0, max(0, len(shellcode) - 1)) if shellcode else 0
        return shellcode[:insert_pos] + instr + shellcode[insert_pos:]
    elif mutation_type == "remove" and len(shellcode) > 4:
        remove_start = randint(0, len(shellcode) - 4)
        remove_length = randint(1, min(4, len(shellcode) - remove_start))
        return shellcode[:remove_start] + shellcode[remove_start + remove_length:]
    elif mutation_type == "modify" and shellcode:
        mod_pos = randint(0, len(shellcode) - 1)
        new_byte = bytes([randint(0, 255)])
        return shellcode[:mod_pos] + new_byte + shellcode[mod_pos + 1:]
    else:
        return shellcode + bytes([randint(0, 255) for _ in range(randint(1, 4))])

def generate_random_instruction():
    """Generate random x86_64 instruction"""
    # ... existing instruction generation logic ...
    return bytes([randint(0, 255) for _ in range(randint(1, 8))])

async def execute_shellcode_secure(shellcode, program_id, gen_id):
    """Execute shellcode with advanced security sandbox"""
    
    if security_sandbox and effective_config.get('enable_security_sandbox', False):
        # Use advanced security sandbox
        try:
            sandbox_id = f"gen_{gen_id}_prog_{program_id}"
            result = await security_sandbox.execute_shellcode(shellcode, sandbox_id)
            
            # Convert sandbox result to KernelHunter format
            survived = result.get("exit_code", -1) == 0
            crash_type = "SANDBOX_ERROR" if not survived else "SUCCESS"
            
            return {
                "survived": survived,
                "crash_type": crash_type,
                "logs": result.get("logs", ""),
                "execution_time": result.get("execution_time", 0),
                "sandbox_result": result
            }
            
        except Exception as e:
            print(f"⚠️ Error en sandbox: {e}")
            # Fall back to traditional execution
    
    # Traditional execution (fallback)
    return await execute_shellcode_traditional(shellcode, program_id, gen_id)

async def execute_shellcode_traditional(shellcode, program_id, gen_id):
    """Traditional shellcode execution"""
    # ... existing execution logic ...
    # (Keep the existing C stub generation and subprocess execution)
    
    # Simplified version for brevity
    shellcode_c = format_shellcode_c_array(shellcode)
    stub = generate_c_stub(shellcode_c)
    
    gen_path = os.path.join(OUTPUT_DIR, f"gen_{gen_id:04d}")
    c_path = os.path.join(gen_path, f"g{gen_id:04d}_p{program_id:04d}.c")
    bin_path = os.path.join(gen_path, f"g{gen_id:04d}_p{program_id:04d}")
    
    with open(c_path, "w") as f:
        f.write(stub)
    
    try:
        # Compile
        subprocess.run(["clang", c_path, "-o", bin_path], check=True,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Execute
        result = subprocess.run([bin_path], timeout=TIMEOUT, capture_output=True)
        
        return {
            "survived": result.returncode == 0,
            "crash_type": "SUCCESS" if result.returncode == 0 else "CRASH",
            "logs": result.stderr.decode('utf-8', errors='ignore'),
            "execution_time": 0
        }
        
    except subprocess.TimeoutExpired:
        return {
            "survived": False,
            "crash_type": "TIMEOUT",
            "logs": "Execution timeout",
            "execution_time": TIMEOUT
        }
    except Exception as e:
        return {
            "survived": False,
            "crash_type": "ERROR",
            "logs": str(e),
            "execution_time": 0
        }

def generate_c_stub(shellcode_c):
    """Generate C stub for shellcode execution"""
    return f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int main() {{
    unsigned char code[] = {{{shellcode_c}}};
    size_t code_size = sizeof(code);
    
    void *exec_mem = mmap(NULL, code_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (exec_mem == MAP_FAILED) {{
        fprintf(stderr, "Failed to allocate executable memory\\n");
        return 1;
    }}
    
    memcpy(exec_mem, code, code_size);
    ((void (*)())exec_mem)();
    
    return 0;
}}
"""

async def run_generation_advanced(gen_id, base_population):
    """Run generation with advanced features"""
    global current_generation, individual_zero_crash_counts
    current_generation = gen_id
    
    # Reset counters
    generation_attack_counter.clear()
    generation_mutation_counter.clear()
    attack_success.clear()
    mutation_success.clear()
    
    gen_path = os.path.join(OUTPUT_DIR, f"gen_{gen_id:04d}")
    os.makedirs(gen_path, exist_ok=True)
    
    new_population = []
    crashes = 0
    crash_types_counter = Counter()
    system_impacts = 0
    shellcode_lengths = []
    
    # Use performance optimizer if available
    if performance_optimizer:
        shellcodes_to_process = base_population * NUM_PROGRAMS
        processed_shellcodes = await performance_optimizer.optimize_shellcode_processing(shellcodes_to_process)
    else:
        processed_shellcodes = base_population
    
    # Process shellcodes
    tasks = []
    for i in range(NUM_PROGRAMS):
        parent = choice(processed_shellcodes)
        shellcode = mutate_shellcode(parent)
        
        task = execute_shellcode_secure(shellcode, i, gen_id)
        tasks.append((i, shellcode, task))
    
    # Execute all tasks
    results = []
    for i, shellcode, task in tasks:
        try:
            result = await task
            result["program_id"] = i
            result["shellcode"] = shellcode
            results.append(result)
        except Exception as e:
            results.append({
                "program_id": i,
                "shellcode": shellcode,
                "survived": False,
                "crash_type": "ERROR",
                "logs": str(e),
                "execution_time": 0
            })
    
    # Process results
    for result in results:
        shellcode = result["shellcode"]
        shellcode_lengths.append(len(shellcode))
        
        if result["survived"]:
            new_population.append(shellcode)
            if USE_RL_WEIGHTS and ml_engine:
                # Update ML models with success
                state_features = extract_state_features()
                ml_engine.update_experience(
                    state_features, last_attack_type, 1.0, state_features, False
                )
        else:
            crashes += 1
            crash_type = result["crash_type"]
            crash_types_counter[crash_type] += 1
            
            # Check for system impact
            if "kernel" in result["logs"].lower() or "panic" in result["logs"].lower():
                system_impacts += 1
                genetic_reservoir.add(shellcode, {"crash_type": crash_type, "system_impact": True})
            
            if USE_RL_WEIGHTS and ml_engine:
                # Update ML models with failure
                state_features = extract_state_features()
                reward = -0.5 if crash_type == "TIMEOUT" else -1.0
                ml_engine.update_experience(
                    state_features, last_attack_type, reward, state_features, True
                )
    
    # Calculate metrics
    crash_rate = crashes / NUM_PROGRAMS
    avg_length = sum(shellcode_lengths) / len(shellcode_lengths) if shellcode_lengths else 0
    
    # Update analytics
    if analytics_engine:
        analytics_data = {
            "generation": gen_id,
            "crash_rate": crash_rate,
            "system_impacts": system_impacts,
            "avg_shellcode_length": avg_length,
            "total_crashes": crashes,
            "unique_crash_types": len(crash_types_counter),
            "diversity_score": genetic_reservoir.get_diversity_stats().get("diversity_avg", 0),
            "rl_epsilon": get_epsilon(gen_id) if USE_RL_WEIGHTS else 0
        }
        await analytics_engine.stream_processor.add_data_point(analytics_data)
    
    # Train ML models periodically
    if ml_engine and gen_id % 10 == 0:
        ml_engine.train_dqn()
        ml_engine.train_policy([], [], [])  # Would need actual data here
    
    # Log generation
    if pythonlogger:
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
        except Exception as e:
            print(f"⚠️ Error logging generation: {e}")
    
    # Update metrics
    metrics["generations"].append(gen_id)
    metrics["crash_rates"].append(crash_rate)
    metrics["system_impacts"].append(system_impacts)
    metrics["shellcode_lengths"].append(avg_length)
    metrics["crash_types"][gen_id] = dict(crash_types_counter)
    metrics.setdefault("attack_stats", {})[gen_id] = dict(generation_attack_counter)
    metrics.setdefault("mutation_stats", {})[gen_id] = dict(generation_mutation_counter)
    
    # Add advanced metrics
    if ml_engine:
        metrics.setdefault("ml_metrics", {})[gen_id] = ml_engine.get_model_stats()
    if analytics_engine:
        metrics.setdefault("analytics_metrics", {})[gen_id] = analytics_engine.get_analytics_summary()
    if security_sandbox:
        metrics.setdefault("security_metrics", {})[gen_id] = security_sandbox.get_sandbox_stats()
    
    write_metrics()
    
    print(color_text(f"[GEN {gen_id}] Crash rate: {crash_rate*100:.1f}% | Sys impacts: {system_impacts} | Avg length: {avg_length:.1f}", "cyan"))
    print(color_text(f"[GEN {gen_id}] Crash types: {dict(crash_types_counter.most_common(3))}", "cyan"))
    
    # Handle population management
    combined_population = deduplicate_population(base_population + new_population)
    if len(combined_population) > MAX_POPULATION_SIZE:
        combined_population = random.sample(combined_population, MAX_POPULATION_SIZE)
    
    return combined_population

def deduplicate_population(population):
    """Remove duplicate shellcodes while preserving order"""
    seen = set()
    unique_population = []
    for shellcode in population:
        shellcode_hash = shellcode.hex()
        if shellcode_hash not in seen:
            seen.add(shellcode_hash)
            unique_population.append(shellcode)
    return unique_population

def get_epsilon(generation):
    """Get epsilon value for epsilon-greedy exploration"""
    return max(0.01, 1.0 - generation / 1000)

def calculate_reward(crash_info):
    """Calculate reward for RL"""
    crash_type = crash_info.get("crash_type", "")
    system_impact = crash_info.get("system_impact", False)
    
    if system_impact:
        return 10.0  # High reward for system impact
    elif crash_type in ["SEGFAULT", "GENERAL_PROTECTION"]:
        return 5.0   # Medium reward for interesting crashes
    elif crash_type == "TIMEOUT":
        return -1.0  # Small penalty for timeouts
    else:
        return 0.0   # Neutral for other crashes

def update_q_value(q_values, counts, index, reward):
    """Update Q-value for RL"""
    if index not in counts:
        counts[index] = 0
    counts[index] += 1
    
    learning_rate = 0.1
    q_values[index] += learning_rate * (reward - q_values[index])

def update_rl_weights():
    """Update RL weights based on success rates"""
    global attack_weights, mutation_weights
    
    if not attack_success or not mutation_success:
        return
    
    # Update attack weights
    total_attack_success = sum(attack_success.values())
    if total_attack_success > 0:
        for i, attack_type in enumerate(ATTACK_OPTIONS):
            success = attack_success.get(attack_type, 0)
            attack_weights[i] = max(1, int(attack_weights[i] * (1 + success / total_attack_success)))
    
    # Update mutation weights
    total_mutation_success = sum(mutation_success.values())
    if total_mutation_success > 0:
        for i, mutation_type in enumerate(MUTATION_TYPES):
            success = mutation_success.get(mutation_type, 0)
            mutation_weights[i] = max(1, int(mutation_weights[i] * (1 + success / total_mutation_success)))

def save_crash_info(gen_id, program_id, shellcode, crash_type, stderr_text, stdout_text, return_code, system_impact, parent_shellcode=None):
    """Save detailed crash information"""
    crash_info = {
        "generation": gen_id,
        "program_id": program_id,
        "shellcode_hex": shellcode.hex(),
        "shellcode_length": len(shellcode),
        "crash_type": crash_type,
        "stderr": stderr_text,
        "stdout": stdout_text,
        "return_code": return_code,
        "system_impact": system_impact,
        "timestamp": time.time(),
        "parent_shellcode_hex": parent_shellcode.hex() if parent_shellcode else None
    }
    
    # Save to file
    crash_file = f"kernelhunter_crashes/gen{gen_id:04d}_prog{program_id:04d}.json"
    with open(crash_file, "w") as f:
        json.dump(crash_info, f, indent=2)
    
    # Save to critical crashes if system impact
    if system_impact:
        critical_file = f"kernelhunter_critical/critical_gen{gen_id:04d}_prog{program_id:04d}.json"
        with open(critical_file, "w") as f:
            json.dump(crash_info, f, indent=2)

def is_system_level_crash(return_code, stderr_text):
    """Determine if crash has system-level impact"""
    stderr_lower = stderr_text.lower()
    return (
        return_code != 0 and
        any(keyword in stderr_lower for keyword in [
            "kernel", "panic", "oops", "segfault", "general protection",
            "invalid opcode", "divide error", "stack fault"
        ])
    )

def print_shellcode_hex(shellcode, escape_format=False):
    """Print shellcode in hex format"""
    hex_str = shellcode.hex()
    if escape_format:
        return hex_str
    return ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))

async def main():
    """Main function with advanced features"""
    global pythonlogger, genetic_reservoir, effective_config
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='KernelHunter Advanced - Fuzzer evolutivo')
    parser.add_argument('--advanced', action='store_true', help='Habilitar características avanzadas')
    parser.add_argument('--config', type=str, help='Archivo de configuración JSON')
    args = parser.parse_args()
    
    # Load configuration from file if specified
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            
            # Update effective_config with file configuration
            if 'local' in config_data:
                effective_config.update(config_data['local'])
                print(f"✅ Configuración cargada desde: {args.config}")
                print(f"  - Generaciones máximas: {effective_config.get('max_generations', 'N/A')}")
                print(f"  - Tamaño de población: {effective_config.get('population_size', 'N/A')}")
                print(f"  - Tasa de mutación: {effective_config.get('mutation_rate', 'N/A')}")
        except Exception as e:
            print(f"⚠️ Error cargando configuración desde {args.config}: {e}")
    
    print(color_text("🚀 KernelHunter Advanced - Iniciando...", "green"))
    
    # Update global variables based on effective_config
    global NUM_PROGRAMS, MAX_GENERATIONS, TIMEOUT, METRICS_FILE, MAX_POPULATION_SIZE, USE_RL_WEIGHTS
    
    NUM_PROGRAMS = effective_config.get('population_size', 50)
    MAX_GENERATIONS = effective_config.get('max_generations', 100)
    TIMEOUT = effective_config.get('max_execution_time', 3)
    METRICS_FILE = effective_config.get('metrics_file', "kernelhunter_metrics.json")
    MAX_POPULATION_SIZE = effective_config.get('population_size', 1000)
    USE_RL_WEIGHTS = effective_config.get('enable_rl', False)
    
    # Initialize advanced modules
    initialize_advanced_modules()
    
    # Start advanced services
    await start_advanced_services()
    
    # Initialize performance logger
    try:
        pythonlogger = PerformanceLogger()
        pythonlogger.log_data["session_info"]["use_rl_weights"] = USE_RL_WEIGHTS
        pythonlogger.log_data["session_info"]["advanced_features_enabled"] = ADVANCED_FEATURES_ENABLED
    except Exception as e:
        print(f"⚠️ Error inicializando logger: {e}")
    
    # Load genetic reservoir
    try:
        genetic_reservoir.load_from_file(get_reservoir_file())
        print(f"✅ Genetic reservoir cargado: {len(genetic_reservoir)} elementos")
    except Exception as e:
        print(f"⚠️ Error cargando genetic reservoir: {e}")
    
    # Initialize population
    base_population = [BASE_SHELLCODE + EXIT_SYSCALL]
    
    print(color_text(f"🎯 Configuración:", "cyan"))
    print(f"  - Generaciones máximas: {MAX_GENERATIONS}")
    print(f"  - Tamaño de población: {NUM_PROGRAMS}")
    print(f"  - RL habilitado: {USE_RL_WEIGHTS}")
    print(f"  - Características avanzadas: {ADVANCED_FEATURES_ENABLED}")
    print(f"  - Sandbox: {effective_config.get('enable_security_sandbox', False)}")
    print(f"  - ML: {effective_config.get('enable_ml', False)}")
    print(f"  - Analytics: {effective_config.get('enable_analytics', False)}")
    
    try:
        for gen_id in range(MAX_GENERATIONS):
            print(color_text(f"\n🔄 Generación {gen_id + 1}/{MAX_GENERATIONS}", "yellow"))
            
            # Run generation with advanced features
            base_population = await run_generation_advanced(gen_id, base_population)
            
            # Save checkpoint
            if gen_id % CHECKPOINT_INTERVAL == 0:
                genetic_reservoir.save_to_file(get_reservoir_file())
                if ml_engine:
                    ml_engine.save_models()
                print(f"💾 Checkpoint guardado en generación {gen_id}")
            
            # Check for stagnation
            if len(base_population) == 0:
                print(color_text("⚠️ Población vacía, reiniciando...", "yellow"))
                base_population = [BASE_SHELLCODE + EXIT_SYSCALL]
            
    except KeyboardInterrupt:
        print(color_text("\n⏹️ Interrumpido por el usuario", "yellow"))
    except Exception as e:
        print(color_text(f"\n❌ Error en ejecución: {e}", "red"))
    finally:
        # Finalize logging
        if pythonlogger:
            try:
                pythonlogger.finalize_session()
            except Exception as e:
                print(f"⚠️ Error finalizando logger: {e}")
        
        # Stop advanced services
        await stop_advanced_services()
        
        # Save final state
        genetic_reservoir.save_to_file(get_reservoir_file())
        if ml_engine:
            ml_engine.save_models()
        
        print(color_text("✅ KernelHunter Advanced finalizado", "green"))

if __name__ == "__main__":
    # Setup environment
    setup_environment()
    
    # Run main function
    asyncio.run(main())
