#!/usr/bin/env python3
"""KernelHunter configuration management module.

This module handles configuration for KernelHunter. The settings can be
stored either globally in ``/etc/kernelhunter/config.json`` or locally in
``~/.config/kernelhunter/config.json``. The configuration contains the path
of the genetic reservoir directory and the OpenAI API key.
"""

import os
import json
from pathlib import Path

class KernelHunterConfig:
    def __init__(self):
        # Modo de ejecución
        self.execution_mode = os.getenv('KH_EXECUTION_MODE', 'local')  # 'local' o 'distributed'
        
        # Configuración local (por defecto)
        self.local_config = {
            'max_generations': 100,
            'population_size': 50,
            'mutation_rate': 0.3,
            'crossover_rate': 0.7,
            'elite_size': 5,
            'stagnation_limit': 20,
            'enable_rl': True,
            'enable_web_interface': True,
            'web_port': 8080,
            'log_level': 'INFO',
            'save_crashes': True,
            'crash_dir': './crashes',
            'metrics_file': './metrics.json',
            'gene_bank_file': './gene_bank.json'
        }
        
        # Configuración distribuida (opcional)
        self.distributed_config = {
            'enable_distributed': False,
            'central_api_url': 'http://localhost:5000',
            'node_id': None,
            'max_workers': 4,
            'sync_interval': 30,
            'heartbeat_interval': 60,
            'crash_upload_enabled': True,
            'metrics_upload_enabled': True
        }
        
        # Configuración de base de datos (solo para modo distribuido)
        self.database_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'kernelhunter'),
            'username': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'pool_size': 10
        }
        
        # Configuración de Linode (solo si se necesita)
        self.linode_config = {
            'api_token': os.getenv('LINODE_API_TOKEN', ''),
            'region': os.getenv('LINODE_REGION', 'us-east'),
            'instance_type': os.getenv('LINODE_TYPE', 'g6-standard-1'),
            'node_count': int(os.getenv('LINODE_NODE_COUNT', 10)),
            'ssh_key_id': os.getenv('LINODE_SSH_KEY_ID', ''),
            'enable_auto_scaling': False
        }
        
        # Cargar configuración desde archivo si existe
        self.load_config()
        
    def load_config(self):
        """Carga configuración desde archivo JSON si existe"""
        config_file = Path('./kernelhunter_config.json')
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    
                # Actualizar configuraciones
                if 'local' in config_data:
                    self.local_config.update(config_data['local'])
                if 'distributed' in config_data:
                    self.distributed_config.update(config_data['distributed'])
                if 'database' in config_data:
                    self.database_config.update(config_data['database'])
                if 'linode' in config_data:
                    self.linode_config.update(config_data['linode'])
                    
                print(f"✅ Configuración cargada desde {config_file}")
            except Exception as e:
                print(f"⚠️ Error cargando configuración: {e}")
    
    def save_config(self):
        """Guarda la configuración actual en archivo JSON"""
        config_data = {
            'local': self.local_config,
            'distributed': self.distributed_config,
            'database': self.database_config,
            'linode': self.linode_config
        }
        
        try:
            with open('./kernelhunter_config.json', 'w') as f:
                json.dump(config_data, f, indent=2)
            print("✅ Configuración guardada en kernelhunter_config.json")
        except Exception as e:
            print(f"⚠️ Error guardando configuración: {e}")
    
    def is_local_mode(self):
        """Verifica si estamos en modo local"""
        return self.execution_mode == 'local'
    
    def is_distributed_mode(self):
        """Verifica si estamos en modo distribuido"""
        return self.execution_mode == 'distributed'
    
    def get_effective_config(self):
        """Obtiene la configuración efectiva según el modo de ejecución"""
        if self.is_local_mode():
            return self.local_config
        else:
            return {**self.local_config, **self.distributed_config}
    
    def setup_local_directories(self):
        """Configura directorios necesarios para ejecución local"""
        directories = [
            self.local_config['crash_dir'],
            './logs',
            './reports',
            './backups'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print("✅ Directorios locales configurados")
    
    def validate_config(self):
        """Valida la configuración actual"""
        errors = []
        
        # Validar configuración local
        if self.local_config['population_size'] < 10:
            errors.append("Tamaño de población debe ser al menos 10")
        
        if not 0 <= self.local_config['mutation_rate'] <= 1:
            errors.append("Tasa de mutación debe estar entre 0 y 1")
        
        if not 0 <= self.local_config['crossover_rate'] <= 1:
            errors.append("Tasa de cruce debe estar entre 0 y 1")
        
        # Validar configuración distribuida si está habilitada
        if self.is_distributed_mode():
            if not self.distributed_config['node_id']:
                errors.append("Node ID es requerido en modo distribuido")
            
            if not self.database_config['password']:
                errors.append("Contraseña de base de datos es requerida en modo distribuido")
        
        if errors:
            print("❌ Errores de configuración:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("✅ Configuración válida")
        return True
    
    def print_config_summary(self):
        """Muestra un resumen de la configuración actual"""
        print("\n" + "="*50)
        print("🔧 CONFIGURACIÓN KERNELHUNTER")
        print("="*50)
        
        print(f"📋 Modo de ejecución: {self.execution_mode.upper()}")
        
        if self.is_local_mode():
            print("\n🏠 CONFIGURACIÓN LOCAL:")
            print(f"  • Generaciones máximas: {self.local_config['max_generations']}")
            print(f"  • Tamaño de población: {self.local_config['population_size']}")
            print(f"  • Tasa de mutación: {self.local_config['mutation_rate']}")
            print(f"  • Tasa de cruce: {self.local_config['crossover_rate']}")
            print(f"  • RL habilitado: {self.local_config['enable_rl']}")
            print(f"  • Interfaz web: {self.local_config['enable_web_interface']}")
            print(f"  • Puerto web: {self.local_config['web_port']}")
        
        if self.is_distributed_mode():
            print("\n🌐 CONFIGURACIÓN DISTRIBUIDA:")
            print(f"  • API central: {self.distributed_config['central_api_url']}")
            print(f"  • Node ID: {self.distributed_config['node_id']}")
            print(f"  • Workers máximos: {self.distributed_config['max_workers']}")
            print(f"  • Sincronización: {self.distributed_config['sync_interval']}s")
        
        print("="*50 + "\n")

# Instancia global de configuración
config = KernelHunterConfig()

# Funciones de conveniencia
def get_config():
    """Obtiene la configuración global"""
    return config

def is_local_mode():
    """Verifica si estamos en modo local"""
    return config.is_local_mode()

def is_distributed_mode():
    """Verifica si estamos en modo distribuido"""
    return config.is_distributed_mode()

def get_effective_config():
    """Obtiene la configuración efectiva"""
    return config.get_effective_config()

def setup_environment():
    """Configura el entorno según el modo de ejecución"""
    config.setup_local_directories()
    
    if not config.validate_config():
        print("❌ Configuración inválida. Abortando...")
        return False
    
    config.print_config_summary()
    return True

# Configuración por defecto para ejecución local
if __name__ == "__main__":
    print("🔧 Configurando KernelHunter...")
    if setup_environment():
        print("✅ KernelHunter configurado correctamente")
        print("\nPara ejecutar en modo local:")
        print("  python KernelHunter.py")
        print("\nPara ejecutar en modo distribuido:")
        print("  KH_EXECUTION_MODE=distributed python KernelHunter.py")
    else:
        print("❌ Error en la configuración")
