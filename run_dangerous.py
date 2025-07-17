#!/usr/bin/env python3
"""
⚠️ KernelHunter - Modo Peligroso
Ejecuta sin sandbox para encontrar vulnerabilidades reales
¡ADVERTENCIA: Puede romper el sistema!
"""

import os
import sys
import subprocess
import time
import signal
import json
from pathlib import Path

def show_warning():
    """Mostrar advertencia de peligro"""
    print("⚠️" * 60)
    print("⚠️                    ADVERTENCIA CRÍTICA                    ⚠️")
    print("⚠️" * 60)
    print("")
    print("🚨 KernelHunter se ejecutará SIN SANDBOX")
    print("🚨 Los shellcodes se ejecutarán DIRECTAMENTE en tu sistema")
    print("🚨 Esto puede causar:")
    print("   - Crashes del kernel")
    print("   - Pérdida de datos")
    print("   - Reinicio del sistema")
    print("   - Daño al hardware (en casos extremos)")
    print("")
    print("🔒 RECOMENDACIONES:")
    print("   - Usar una máquina virtual dedicada")
    print("   - Hacer backup completo del sistema")
    print("   - No ejecutar en producción")
    print("   - Tener un plan de recuperación")
    print("")
    
    response = input("¿Estás seguro de que quieres continuar? (escribe 'SI, SOY PELIGROSO'): ")
    if response != "SI, SOY PELIGROSO":
        print("❌ Ejecución cancelada por seguridad")
        sys.exit(1)
    
    print("")
    print("🔥 ¡MODO PELIGROSO ACTIVADO!")
    print("🔥 KernelHunter buscará vulnerabilidades REALES")
    print("")

def setup_dangerous_environment():
    """Configurar entorno para ejecución peligrosa"""
    print("🔥 Configurando entorno para ejecución peligrosa...")
    
    # Crear directorios
    os.makedirs("kernelhunter_generations", exist_ok=True)
    os.makedirs("crashes", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("critical_crashes", exist_ok=True)
    
    # Variables de entorno para modo peligroso
    env_vars = {
        "KH_EXECUTION_MODE": "dangerous",
        "KH_LOG_LEVEL": "INFO",
        "KH_DANGEROUS_MODE": "true",
        "KH_DISABLE_SANDBOX": "true",
        "KH_ENABLE_REAL_CRASHES": "true",
        "MAX_EXECUTION_TIME": "10",
        "MAX_MEMORY_MB": "2048"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("✅ Entorno configurado para ejecución peligrosa")

def run_kernelhunter_dangerous():
    """Ejecutar KernelHunter en modo peligroso"""
    print("🔥 Iniciando KernelHunter en modo peligroso...")
    
    # Configuración para encontrar vulnerabilidades reales
    config = {
        "max_generations": 5000,
        "population_size": 200,
        "mutation_rate": 0.4,
        "crossover_rate": 0.7,
        "elite_size": 15,
        "stagnation_limit": 50,
        "enable_rl": True,
        "enable_ml": True,
        "enable_analytics": True,
        "enable_security_sandbox": False,
        "sandbox_isolation_level": "none",
        "max_execution_time": 10,
        "max_memory_mb": 2048,
        "log_level": "INFO"
    }
    
    # Guardar configuración
    with open("dangerous_config_temp.json", "w") as f:
        json.dump({"local": config}, f, indent=2)
    
    try:
        # Ejecutar KernelHunter sin sandbox
        cmd = [
            "python3", "kernelHunter.py",
            "--config", "dangerous_config_temp.json"
        ]
        
        print("🎯 Ejecutando con configuración peligrosa:")
        print(f"  - Generaciones: {config['max_generations']}")
        print(f"  - Población: {config['population_size']}")
        print(f"  - ML/RL: Habilitado")
        print(f"  - Analytics: Habilitado")
        print(f"  - Sandbox: DESHABILITADO ⚠️")
        print(f"  - Timeout: {config['max_execution_time']}s")
        print(f"  - Memoria: {config['max_memory_mb']}MB")
        
        start_time = time.time()
        process = subprocess.Popen(cmd)
        
        # Esperar a que termine
        process.wait()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Ejecución completada en {duration:.2f} segundos")
        print("🔥 Revisa la carpeta 'crashes' y 'critical_crashes' para vulnerabilidades encontradas")
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrumpido por el usuario")
        process.terminate()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Limpiar archivo temporal
        if os.path.exists("dangerous_config_temp.json"):
            os.remove("dangerous_config_temp.json")

def show_recovery_tips():
    """Mostrar consejos de recuperación"""
    print("\n💡 Si el sistema se rompe:")
    print("  - Reinicia la máquina")
    print("  - Usa modo de recuperación si es necesario")
    print("  - Restaura desde backup si tienes")
    print("  - Revisa logs del kernel: dmesg | tail -50")
    print("  - Verifica integridad del sistema")

def main():
    """Función principal"""
    print("🔥 KernelHunter - Modo Peligroso")
    print("=" * 50)
    
    # Mostrar advertencia
    show_warning()
    
    # Verificar que estamos en un entorno seguro
    print("🔍 Verificando entorno...")
    
    # Verificar si estamos en una VM
    try:
        with open("/sys/class/dmi/id/product_name", "r") as f:
            product = f.read().strip().lower()
            if "virtual" in product or "vmware" in product or "kvm" in product:
                print("✅ Ejecutando en entorno virtual (más seguro)")
            else:
                print("⚠️ Ejecutando en hardware real (más peligroso)")
    except:
        print("⚠️ No se pudo determinar el entorno")
    
    # Configurar entorno
    setup_dangerous_environment()
    
    # Mostrar consejos de recuperación
    show_recovery_tips()
    
    # Ejecutar
    run_kernelhunter_dangerous()

if __name__ == "__main__":
    main() 