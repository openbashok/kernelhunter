#!/usr/bin/env python3
"""
Script de ejecución rápida para KernelHunter
Optimizado para velocidad máxima sin ML/RL
"""

import os
import sys
import subprocess
import time
import signal
import asyncio
import json
from pathlib import Path

def setup_fast_environment():
    """Configurar entorno para ejecución rápida"""
    print("⚡ Configurando entorno para velocidad máxima...")
    
    # Crear directorios mínimos
    os.makedirs("kernelhunter_generations", exist_ok=True)
    os.makedirs("crashes", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Variables de entorno para velocidad
    env_vars = {
        "KH_EXECUTION_MODE": "local",
        "KH_LOG_LEVEL": "WARNING",
        "KH_FAST_MODE": "true",
        "KH_DISABLE_ML": "true",
        "KH_DISABLE_ANALYTICS": "true",
        "KH_DISABLE_SANDBOX": "true",
        "MAX_EXECUTION_TIME": "5",
        "MAX_MEMORY_MB": "256"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("✅ Entorno configurado para velocidad máxima")

def run_kernelhunter_fast():
    """Ejecutar KernelHunter en modo rápido"""
    print("🚀 Iniciando KernelHunter en modo rápido...")
    
    # Configuración optimizada para velocidad
    config = {
        "max_generations": 10000,
        "population_size": 500,
        "mutation_rate": 0.5,
        "crossover_rate": 0.8,
        "elite_size": 20,
        "stagnation_limit": 100,
        "enable_rl": False,
        "enable_ml": False,
        "enable_analytics": False,
        "enable_security_sandbox": False,
        "max_execution_time": 5,
        "max_memory_mb": 256,
        "log_level": "WARNING"
    }
    
    # Guardar configuración temporal
    with open("fast_config_temp.json", "w") as f:
        json.dump({"local": config}, f, indent=2)
    
    try:
        # Ejecutar KernelHunter con configuración rápida
        cmd = [
            "python3", "kernelHunter.py",
            "--config", "fast_config_temp.json"
        ]
        
        print("🎯 Ejecutando con configuración optimizada:")
        print(f"  - Generaciones: {config['max_generations']}")
        print(f"  - Población: {config['population_size']}")
        print(f"  - ML/RL: Deshabilitado")
        print(f"  - Analytics: Deshabilitado")
        print(f"  - Sandbox: Deshabilitado")
        print(f"  - Timeout: {config['max_execution_time']}s")
        
        start_time = time.time()
        process = subprocess.Popen(cmd)
        
        # Esperar a que termine
        process.wait()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Ejecución completada en {duration:.2f} segundos")
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrumpido por el usuario")
        process.terminate()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Limpiar archivo temporal
        if os.path.exists("fast_config_temp.json"):
            os.remove("fast_config_temp.json")

def show_performance_tips():
    """Mostrar consejos de rendimiento"""
    print("\n💡 Consejos para máxima velocidad:")
    print("  - Usar CPU con muchos cores")
    print("  - Tener suficiente RAM (8GB+)")
    print("  - Usar SSD para I/O rápido")
    print("  - Cerrar aplicaciones innecesarias")
    print("  - Considerar ejecutar en modo distribuido")

def main():
    """Función principal"""
    print("⚡ KernelHunter - Modo Rápido")
    print("=" * 40)
    
    # Verificar dependencias básicas
    try:
        import subprocess
        result = subprocess.run(["clang", "--version"], capture_output=True)
        if result.returncode != 0:
            print("⚠️ Clang no encontrado. Instalando...")
            subprocess.run(["sudo", "apt-get", "install", "-y", "clang"], check=True)
    except:
        print("⚠️ Error verificando compilador")
    
    # Configurar entorno
    setup_fast_environment()
    
    # Mostrar consejos
    show_performance_tips()
    
    # Ejecutar
    run_kernelhunter_fast()

if __name__ == "__main__":
    main() 