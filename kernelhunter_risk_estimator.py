#!/usr/bin/env python3

import subprocess
import json
import sys
import re

# Ejecuta comandos externos y devuelve el resultado
def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout + result.stderr

# Evalúa el nivel de riesgo basado en reglas predefinidas
def evaluar_riesgo(output):
    riesgo = 0

    # Excepción del CPU (GDB, kernel logs)
    if re.search(r'(general protection|invalid opcode|kernel panic|segfault.*kernel)', output, re.I):
        riesgo += 40

    # Syscalls críticas
    if re.search(r'\b(mprotect|mmap|ptrace|ioctl|execve)\b', output):
        riesgo += 20

    # Acceso a memoria kernel-space
    if re.search(r'0xffff[0-9a-fA-F]{4,}', output):
        riesgo += 30

    # Eventos críticos kernel
    if 'kernel panic' in output.lower():
        riesgo += 100
    elif 'stack smashing' in output.lower():
        riesgo += 50

    return riesgo

# Clasifica el riesgo según puntaje
def clasificar_riesgo(puntaje):
    if puntaje >= 100:
        return 'Crítico (kernel exploit potencial)'
    elif puntaje >= 60:
        return 'Alto'
    elif puntaje >= 30:
        return 'Medio'
    else:
        return 'Bajo (probablemente user-space)'

# Análisis principal
def analizar_binario(path_binario):
    resultados = {}

    comandos = {
        'GDB': f'gdb --batch -ex "run" -ex "bt" {path_binario}',
        'Valgrind': f'valgrind --leak-check=full {path_binario}',
        'Strace': f'strace -ff -o strace_output {path_binario}',
        'Ltrace': f'ltrace {path_binario}',
        'Kernel Logs': 'dmesg | tail -n 50'
    }

    output_total = ""
    for nombre, cmd in comandos.items():
        print(f"Ejecutando {nombre}...")
        output = run_cmd(cmd)
        resultados[nombre] = output
        output_total += output

    puntaje = evaluar_riesgo(output_total)
    nivel_riesgo = clasificar_riesgo(puntaje)

    reporte = {
        'binario': path_binario,
        'puntaje_riesgo': puntaje,
        'nivel_riesgo': nivel_riesgo,
        'detalle': resultados
    }

    print(json.dumps(reporte, indent=4))

# Punto de entrada
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: ./kernelhunter_risk_estimator.py <ruta_al_binario>")
        sys.exit(1)

    analizar_binario(sys.argv[1])
