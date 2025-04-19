#!/usr/bin/env python3
# KernelHunter - Fuzzer evolutivo para encontrar vulnerabilidades en el sistema operativo
# 
# Un enfoque evolutivo para generar shellcodes que potencialmente puedan impactar
# la estabilidad del sistema operativo.

import os
import subprocess
import random
import time
import signal
import sys
import json
from random import randint, choice
from collections import Counter

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
NUM_PROGRAMS = 20
MAX_GENERATIONS = 10000
TIMEOUT = 3
LOG_FILE = "kernelhunter_survivors.txt"
CRASH_LOG = "kernelhunter_crashes.txt"
METRICS_FILE = "kernelhunter_metrics.json"
CHECKPOINT_INTERVAL = 10

# Crear directorios necesarios
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("kernelhunter_crashes", exist_ok=True)
os.makedirs("kernelhunter_critical", exist_ok=True)

# Métricas para seguimiento
metrics = {
    "generations": [],
    "crash_rates": [],
    "crash_types": {},
    "system_impacts": [],
    "shellcode_lengths": []
}

def generate_random_instruction():
    """Genera una instrucción aleatoria con mayor probabilidad de instrucciones interesantes"""
    # Opciones para generar instrucciones
    options = [
        "random_bytes",      # bytes completamente aleatorios
        "syscall_setup",     # configuración para syscall
        "syscall",           # instrucción de syscall
        "memory_access",     # acceso a memoria potencialmente peligroso
        "privileged",        # instrucciones privilegiadas
    ]
    
    weights = [50, 25, 10, 10, 5]  # Probabilidades relativas
    choice_type = random.choices(options, weights=weights)[0]
    
    if choice_type == "random_bytes":
        instr_length = randint(1, 6)
        return bytes([randint(0, 255) for _ in range(instr_length)])
    
    elif choice_type == "syscall_setup":
        return random.choice(SYSCALL_SETUP) + bytes([randint(0, 255) for _ in range(randint(1, 4))])
    
    elif choice_type == "syscall":
        return SYSCALL_PATTERN
    
    elif choice_type == "memory_access":
        # Instrucciones que acceden a memoria, más probabilidad de fallos
        mem_instructions = [
            b"\x48\x8b", # mov reg, [reg]
            b"\x48\x89", # mov [reg], reg
            b"\xff",     # varios opcodes que pueden acceder/modificar memoria
            b"\x0f\xae", # instrucciones MFENCE, LFENCE, SFENCE, CLFLUSH
        ]
        instr = random.choice(mem_instructions)
        # Añadir bytes aleatorios como operandos
        return instr + bytes([randint(0, 255) for _ in range(randint(1, 3))])
    
    elif choice_type == "privileged":
        # Instrucciones privilegiadas o que podrían causar excepciones
        privileged_instructions = [
            b"\x0f\x01", # Varias instrucciones de sistema (SIDT, SGDT, etc.)
            b"\xcd\x80", # int 0x80 (syscall de 32-bit)
            b"\xf4",     # HLT (halt)
            b"\x0f\x00", # Instrucciones de control de segmentos
            b"\x0f\x06", # CLTS (clear task switched flag)
            b"\x0f\x09", # WBINVD (write back and invalidate cache)
        ]
        return random.choice(privileged_instructions)
    
    # Por defecto
    return bytes([randint(0, 255) for _ in range(randint(1, 4))])

def mutate_shellcode(shellcode, mutation_rate=0.8):
    """Muta el shellcode con diferentes estrategias"""
    if shellcode.endswith(EXIT_SYSCALL):
        core = shellcode[:-len(EXIT_SYSCALL)]
    else:
        core = shellcode
    
    mutation_type = random.choices([
        "add",      # Añadir instrucción
        "remove",   # Quitar instrucción
        "modify",   # Modificar instrucción existente
        "duplicate" # Duplicar sección
    ], weights=[70, 10, 15, 5])[0]
    
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
    system_level_signals = [
        signal.SIGBUS,   # Error de bus (acceso a hardware)
        signal.SIGSYS,   # Syscall inválido
        signal.SIGILL,   # Instrucción ilegal
        signal.SIGABRT,  # Abort - podría ser del sistema
    ]
    
    if return_code < 0 and -return_code in [s.value for s in system_level_signals]:
        return True
    
    # Buscar mensajes de error del kernel en stderr
    kernel_indicators = ["kernel", "panic", "oops", "BUG:", "WARNING:", "general protection"]
    if any(indicator in stderr.lower() for indicator in kernel_indicators):
        return True
    
    return False

def save_crash_info(gen_id, prog_id, shellcode, crash_type, stderr, stdout, return_code, is_system_impact):
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
        "timestamp": time.time()
    }
    
    with open(f"{save_dir}/crash_gen{gen_id:04d}_prog{prog_id:04d}.json", "w") as f:
        json.dump(crash_info, f, indent=2)

def run_generation(gen_id, base_population):
    """Ejecuta una generación completa del algoritmo evolutivo"""
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
            # Seleccionar un padre (preferencia por shellcodes más cortos)
            if random.random() < 0.2 and len(base_population) > 1:
                # Seleccionar uno de los shellcodes más cortos 20% del tiempo
                sorted_by_len = sorted(base_population, key=len)
                parent = sorted_by_len[randint(0, min(5, len(sorted_by_len)-1))]
            else:
                parent = choice(base_population)
            
            shellcode = mutate_shellcode(parent)
            shellcode_c = format_shellcode_c_array(shellcode)
            shellcode_lengths.append(len(shellcode))

            # Generar el programa C con el shellcode
            stub = f"""
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>
#include <unistd.h>

unsigned char code[] = {{{shellcode_c}}};

int main() {{
    // Desenfocar stdout/stderr para capturar mensajes del kernel
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
    
    return 0;
}}
"""

            c_path = os.path.join(gen_path, f"prog_{i:04d}.c")
            bin_path = os.path.join(gen_path, f"prog_{i:04d}")

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

                if result.returncode == 0:
                    # El shellcode sobrevivió
                    new_population.append(shellcode)
                    log.write(f"Survivor {i:04d}: {shellcode.hex()} (len: {len(shellcode)})\n")
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
                        msg = f"Crash {i:04d}: {sig_name} | Shellcode: {shellcode.hex()}"
                        
                        # Registrar el tipo de crash
                        crash_types_counter[crash_type] += 1
                        
                    else:
                        # Salida con código de error
                        crash_type = f"EXIT_{result.returncode}"
                        msg = f"Crash {i:04d}: exit code {result.returncode} | Shellcode: {shellcode.hex()}"
                        crash_types_counter[crash_type] += 1
                    
                    # Marcar impactos del sistema
                    if is_system_impact:
                        msg = f"[SYSTEM IMPACT] {msg}"
                        system_impacts += 1
                    
                    print(msg)
                    crash_log.write(msg + "\n")
                    
                    # Guardar el código fuente y la información del crash
                    with open(f"kernelhunter_crashes/gen{gen_id:04d}_prog{i:04d}.c", "w") as fc:
                        fc.write(stub)
                    
                    # Guardar información detallada del crash
                    save_crash_info(gen_id, i, shellcode, crash_type, stderr_text, 
                                   stdout_text, result.returncode, is_system_impact)
                    
                    crashes += 1

            # Manejar otros tipos de errores
            except subprocess.TimeoutExpired:
                crash_type = "TIMEOUT"
                msg = f"Crash {i:04d}: TIMEOUT | Shellcode: {shellcode.hex()}"
                print(msg)
                crash_log.write(msg + "\n")
                with open(f"kernelhunter_crashes/gen{gen_id:04d}_prog{i:04d}.c", "w") as fc:
                    fc.write(stub)
                crashes += 1
                crash_types_counter[crash_type] += 1
                
            except subprocess.CalledProcessError:
                crash_type = "COMPILE_ERROR"
                msg = f"Crash {i:04d}: COMPILATION ERROR | Shellcode: {shellcode.hex()}"
                print(msg)
                crash_log.write(msg + "\n")
                crashes += 1
                crash_types_counter[crash_type] += 1

    # Asegurar que siempre hay población
    if not new_population:
        # Si todos crashearon, reiniciar con la población base pero con algunos modificados
        print(f"[GEN {gen_id}] Todos crashearon. Reiniciando población.")
        new_population = [BASE_SHELLCODE + EXIT_SYSCALL]
        for _ in range(min(5, len(base_population))):
            parent = choice(base_population)
            simplified = parent[:len(parent)//2] + EXIT_SYSCALL if len(parent) > 10 else parent
            new_population.append(simplified)

    # Calcular y mostrar métricas
    crash_rate = crashes / NUM_PROGRAMS
    avg_length = sum(shellcode_lengths) / len(shellcode_lengths) if shellcode_lengths else 0
    
    print(f"[GEN {gen_id}] Crash rate: {crash_rate*100:.1f}% | Sys impacts: {system_impacts} | Avg length: {avg_length:.1f}")
    print(f"[GEN {gen_id}] Crash types: {dict(crash_types_counter.most_common(3))}")
    
    # Actualizar métricas
    metrics["generations"].append(gen_id)
    metrics["crash_rates"].append(crash_rate)
    metrics["system_impacts"].append(system_impacts)
    metrics["shellcode_lengths"].append(avg_length)
    metrics["crash_types"][gen_id] = dict(crash_types_counter)
    
    # Guardar checkpoint periódico
    if gen_id % CHECKPOINT_INTERVAL == 0:
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)

    return new_population

def analyze_crash_files():
    """Analiza los archivos de crash guardados para buscar patrones"""
    crash_dir = "kernelhunter_critical"
    crash_files = [f for f in os.listdir(crash_dir) if f.endswith(".json")]
    
    if not crash_files:
        print("No se encontraron crashes críticos para analizar.")
        return
    
    print(f"\nAnalizando {len(crash_files)} crashes con potencial impacto a nivel de sistema...")
    
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
    
    print(f"\nResumen de análisis:")
    print(f"- Promedio de longitud de shellcode: {avg_length:.1f} bytes")
    print(f"- Señales más comunes: {dict(crash_signals.most_common(3))}")
    
    if shellcode_segments:
        print(f"- Segmentos de shellcode más comunes:")
        for segment, count in shellcode_segments.most_common(5):
            if count > 1:
                print(f"  {segment}: {count} apariciones")

def main():
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
    print(f"Configuración: {NUM_PROGRAMS} programas por generación, timeout: {TIMEOUT}s")
    
    initial = BASE_SHELLCODE + EXIT_SYSCALL
    population = [initial]
    
    with open(LOG_FILE, "w") as log:
        log.write("KernelHunter - Log de shellcodes sobrevivientes\n=======================================\n")
    with open(CRASH_LOG, "w") as crash_log:
        crash_log.write("KernelHunter - Log de crashes clasificados\n=======================================\n")
    
    try:
        for gen in range(MAX_GENERATIONS):
            print(f"\nGeneración {gen}/{MAX_GENERATIONS} (población: {len(population)})")
            population = run_generation(gen, population)
            time.sleep(0.2)
    
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario.")
    
    finally:
        print("\nGuardando métricas finales...")
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("\nEvolución completada.")
        analyze_crash_files()
        
        print("\nResumen final:")
        print(f"- Total de generaciones: {len(metrics['generations'])}")
        if metrics['system_impacts']:
            total_impacts = sum(metrics['system_impacts'])
            print(f"- Total de impactos potenciales al sistema: {total_impacts}")
            print(f"- Directorio de crashes críticos: kernelhunter_critical/")
        
        print("\nKernelHunter ha completado su ejecución.")

if __name__ == "__main__":
    main()

