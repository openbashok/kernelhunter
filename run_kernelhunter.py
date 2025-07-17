#!/usr/bin/env python3
"""
KernelHunter Launcher
Script para ejecutar KernelHunter de forma fácil en modo local o distribuido
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

def print_banner():
    """Muestra el banner de KernelHunter"""
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

def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    required_modules = [
        'asyncio', 'concurrent.futures', 'multiprocessing',
        'json', 'random', 'time', 'signal', 'subprocess'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Módulos faltantes: {', '.join(missing_modules)}")
        print("💡 Instala las dependencias con: pip install -r requirements.txt")
        return False
    
    print("✅ Todas las dependencias están disponibles")
    return True

def create_default_config():
    """Crea una configuración por defecto si no existe"""
    config_file = Path('./kernelhunter_config.json')
    
    if not config_file.exists():
        default_config = {
            "local": {
                "max_generations": 100,
                "population_size": 50,
                "mutation_rate": 0.3,
                "crossover_rate": 0.7,
                "elite_size": 5,
                "stagnation_limit": 20,
                "enable_rl": True,
                "enable_web_interface": True,
                "web_port": 8080,
                "log_level": "INFO",
                "save_crashes": True,
                "crash_dir": "./crashes",
                "metrics_file": "./metrics.json",
                "gene_bank_file": "./gene_bank.json"
            },
            "distributed": {
                "enable_distributed": False,
                "central_api_url": "http://localhost:5000",
                "node_id": None,
                "max_workers": 4,
                "sync_interval": 30,
                "heartbeat_interval": 60,
                "crash_upload_enabled": True,
                "metrics_upload_enabled": True
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "kernelhunter",
                "username": "postgres",
                "password": "",
                "pool_size": 10
            },
            "linode": {
                "api_token": "",
                "region": "us-east",
                "instance_type": "g6-standard-1",
                "node_count": 10,
                "ssh_key_id": "",
                "enable_auto_scaling": False
            }
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"✅ Configuración por defecto creada en {config_file}")
            return True
        except Exception as e:
            print(f"❌ Error creando configuración: {e}")
            return False
    
    return True

def setup_local_environment():
    """Configura el entorno para ejecución local"""
    print("🏠 Configurando entorno local...")
    
    # Crear directorios necesarios
    directories = [
        './crashes',
        './logs',
        './reports',
        './backups',
        './kernelhunter_generations'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Entorno local configurado")

def setup_distributed_environment():
    """Configura el entorno para ejecución distribuida"""
    print("🌐 Configurando entorno distribuido...")
    
    # Verificar variables de entorno necesarias
    required_env_vars = ['DB_HOST', 'DB_PASSWORD', 'LINODE_API_TOKEN']
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Variables de entorno faltantes: {', '.join(missing_vars)}")
        print("💡 Configura las variables de entorno necesarias:")
        for var in missing_vars:
            print(f"   export {var}=<valor>")
        return False
    
    print("✅ Entorno distribuido configurado")
    return True

def run_kernelhunter(mode='local', use_rl=False, config_file=None):
    """Ejecuta KernelHunter con la configuración especificada"""
    
    # Configurar variables de entorno
    os.environ['KH_EXECUTION_MODE'] = mode
    
    # Construir comando
    cmd = [sys.executable, 'KernelHunter.py']
    
    if use_rl:
        cmd.append('--use-rl-weights')
    
    if mode == 'local':
        cmd.append('--local')
    elif mode == 'distributed':
        cmd.append('--distributed')
    
    if config_file:
        cmd.extend(['--config', config_file])
    
    print(f"🚀 Ejecutando: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # Ejecutar KernelHunter
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando KernelHunter: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Ejecución interrumpida por el usuario")
        return False

def show_status():
    """Muestra el estado actual del sistema"""
    print("\n📊 ESTADO DEL SISTEMA")
    print("=" * 30)
    
    # Verificar archivos importantes
    files_to_check = [
        ('KernelHunter.py', 'Script principal'),
        ('kernelhunter_config.py', 'Configuración'),
        ('kernelhunter_config.json', 'Configuración JSON'),
        ('./crashes', 'Directorio de crashes'),
        ('./logs', 'Directorio de logs')
    ]
    
    for file_path, description in files_to_check:
        if Path(file_path).exists():
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ {description}: {file_path} (faltante)")
    
    # Verificar configuración
    try:
        from kernelhunter_config import get_config
        config = get_config()
        print(f"📋 Modo de ejecución: {config.execution_mode.upper()}")
        print(f"🏠 Población local: {config.local_config['population_size']}")
        print(f"🔄 Generaciones máx: {config.local_config['max_generations']}")
    except Exception as e:
        print(f"⚠️ Error cargando configuración: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="KernelHunter Launcher - Fuzzer evolutivo para vulnerabilidades del SO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python run_kernelhunter.py                    # Ejecutar en modo local
  python run_kernelhunter.py --local            # Forzar modo local
  python run_kernelhunter.py --distributed      # Ejecutar en modo distribuido
  python run_kernelhunter.py --rl               # Habilitar RL
  python run_kernelhunter.py --status           # Mostrar estado
  python run_kernelhunter.py --setup-local      # Configurar entorno local
        """
    )
    
    parser.add_argument('--local', action='store_true',
                        help='Ejecutar en modo local (por defecto)')
    parser.add_argument('--distributed', action='store_true',
                        help='Ejecutar en modo distribuido')
    parser.add_argument('--rl', action='store_true',
                        help='Habilitar reinforcement learning')
    parser.add_argument('--config', type=str,
                        help='Archivo de configuración personalizado')
    parser.add_argument('--status', action='store_true',
                        help='Mostrar estado del sistema')
    parser.add_argument('--setup-local', action='store_true',
                        help='Configurar entorno local')
    parser.add_argument('--check-deps', action='store_true',
                        help='Verificar dependencias')
    parser.add_argument('--create-config', action='store_true',
                        help='Crear configuración por defecto')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Comandos de utilidad
    if args.check_deps:
        check_dependencies()
        return
    
    if args.create_config:
        create_default_config()
        return
    
    if args.status:
        show_status()
        return
    
    if args.setup_local:
        setup_local_environment()
        return
    
    # Verificar dependencias antes de ejecutar
    if not check_dependencies():
        return
    
    # Crear configuración por defecto si no existe
    if not create_default_config():
        return
    
    # Determinar modo de ejecución
    mode = 'local'
    if args.distributed:
        mode = 'distributed'
    elif args.local:
        mode = 'local'
    
    # Configurar entorno según el modo
    if mode == 'local':
        setup_local_environment()
    elif mode == 'distributed':
        if not setup_distributed_environment():
            print("❌ No se pudo configurar el entorno distribuido")
            return
    
    # Ejecutar KernelHunter
    success = run_kernelhunter(
        mode=mode,
        use_rl=args.rl,
        config_file=args.config
    )
    
    if success:
        print("\n✅ KernelHunter completado exitosamente")
    else:
        print("\n❌ KernelHunter falló")
        sys.exit(1)

if __name__ == "__main__":
    main() 