#!/usr/bin/env python3
"""
Script de ejecución avanzado para KernelHunter
Integra todas las características avanzadas y proporciona una interfaz unificada
"""

import os
import sys
import argparse
import asyncio
import json
import signal
import subprocess
import time
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kernelhunter_advanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KernelHunterAdvanced:
    """Clase principal para ejecutar KernelHunter con características avanzadas"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "kernelhunter_config.json"
        self.config = self.load_config()
        self.processes = []
        self.services = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Cargar configuración"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Archivo de configuración {self.config_path} no encontrado, usando configuración por defecto")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Obtener configuración por defecto"""
        return {
            "local": {
                "max_generations": 1000,
                "population_size": 100,
                "mutation_rate": 0.3,
                "crossover_rate": 0.7,
                "elite_size": 10,
                "stagnation_limit": 50,
                "enable_rl": True,
                "enable_web_interface": True,
                "web_port": 8080,
                "log_level": "INFO",
                "save_crashes": True,
                "crash_dir": "./crashes",
                "metrics_file": "./metrics.json",
                "gene_bank_file": "./gene_bank.json",
                            "enable_advanced_features": True,
            "enable_ml": True,
            "enable_analytics": True,
            "enable_security_sandbox": False,  # ¡SIN SANDBOX por defecto!
            "sandbox_isolation_level": "none",  # Sin aislamiento
                "max_execution_time": 30,
                "max_memory_mb": 1024
            },
            "distributed": {
                "enable_distributed": False,
                "central_api_url": "http://localhost:5000",
                "node_id": None,
                "max_workers": 8,
                "sync_interval": 30,
                "heartbeat_interval": 60,
                "crash_upload_enabled": True,
                "metrics_upload_enabled": True
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "kernelhunter",
                "username": "kernelhunter",
                "password": "",
                "pool_size": 20
            },
            "monitoring": {
                "prometheus_port": 9090,
                "grafana_port": 3000,
                "enable_telemetry": True,
                "log_retention_days": 30
            }
        }
    
    def setup_environment(self):
        """Configurar entorno de ejecución"""
        logger.info("🔧 Configurando entorno...")
        
        # Crear directorios necesarios
        directories = [
            "logs",
            "data",
            "crashes",
            "models",
            "metrics",
            "dashboards",
            "reports"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Configurar variables de entorno
        env_vars = {
            "KH_EXECUTION_MODE": "local",
            "KH_LOG_LEVEL": self.config["local"]["log_level"],
            "DB_HOST": self.config["database"]["host"],
            "DB_PORT": str(self.config["database"]["port"]),
            "DB_NAME": self.config["database"]["database"],
            "DB_USER": self.config["database"]["username"],
            "DB_PASSWORD": self.config["database"]["password"],
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "PROMETHEUS_PORT": str(self.config["monitoring"]["prometheus_port"]),
            "GRAFANA_PORT": str(self.config["monitoring"]["grafana_port"]),
            "SANDBOX_ISOLATION_LEVEL": self.config["local"]["sandbox_isolation_level"],
            "MAX_EXECUTION_TIME": str(self.config["local"]["max_execution_time"]),
            "MAX_MEMORY_MB": str(self.config["local"]["max_memory_mb"])
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        logger.info("✅ Entorno configurado")
    
    async def start_database_services(self):
        """Iniciar servicios de base de datos"""
        logger.info("🗄️ Iniciando servicios de base de datos...")
        
        try:
            # Iniciar PostgreSQL
            if not self.is_service_running("postgresql"):
                subprocess.run(["sudo", "systemctl", "start", "postgresql"], check=True)
                logger.info("✅ PostgreSQL iniciado")
            
            # Iniciar Redis
            if not self.is_service_running("redis"):
                subprocess.run(["sudo", "systemctl", "start", "redis"], check=True)
                logger.info("✅ Redis iniciado")
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Error iniciando servicios de base de datos: {e}")
    
    async def start_monitoring_services(self):
        """Iniciar servicios de monitoreo"""
        if not self.config["monitoring"]["enable_telemetry"]:
            return
        
        logger.info("📊 Iniciando servicios de monitoreo...")
        
        try:
            # Iniciar Prometheus
            prometheus_cmd = [
                "docker", "run", "-d", "--name", "prometheus",
                "-p", f"{self.config['monitoring']['prometheus_port']}:9090",
                "-v", f"{os.getcwd()}/prometheus.yml:/etc/prometheus/prometheus.yml",
                "prom/prometheus"
            ]
            subprocess.run(prometheus_cmd, check=True)
            logger.info("✅ Prometheus iniciado")
            
            # Iniciar Grafana
            grafana_cmd = [
                "docker", "run", "-d", "--name", "grafana",
                "-p", f"{self.config['monitoring']['grafana_port']}:3000",
                "-v", f"{os.getcwd()}/dashboards:/var/lib/grafana/dashboards",
                "grafana/grafana"
            ]
            subprocess.run(grafana_cmd, check=True)
            logger.info("✅ Grafana iniciado")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Error iniciando servicios de monitoreo: {e}")
    
    def is_service_running(self, service_name: str) -> bool:
        """Verificar si un servicio está ejecutándose"""
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service_name],
                capture_output=True,
                text=True
            )
            return result.stdout.strip() == "active"
        except:
            return False
    
    async def start_web_interface(self):
        """Iniciar interfaz web"""
        if not self.config["local"]["enable_web_interface"]:
            return
        
        logger.info("🌐 Iniciando interfaz web...")
        
        try:
            # Verificar si el puerto está disponible
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', self.config["local"]["web_port"]))
            sock.close()
            
            if result == 0:
                logger.info(f"✅ Interfaz web ya ejecutándose en puerto {self.config['local']['web_port']}")
                return
            
            # Iniciar servidor web
            web_cmd = [
                "python3", "-m", "http.server",
                str(self.config["local"]["web_port"]),
                "--directory", "web_interface"
            ]
            
            process = subprocess.Popen(web_cmd)
            self.processes.append(process)
            
            # Esperar a que el servidor esté listo
            await asyncio.sleep(2)
            logger.info(f"✅ Interfaz web iniciada en puerto {self.config['local']['web_port']}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error iniciando interfaz web: {e}")
    
    async def start_advanced_services(self):
        """Iniciar servicios avanzados"""
        logger.info("🚀 Iniciando servicios avanzados...")
        
        try:
            # Importar módulos avanzados
            from advanced_ml_engine import get_ml_engine
            from performance_optimizer import get_performance_optimizer
            from real_time_analytics import get_analytics_engine
            from advanced_security_sandbox import get_security_sandbox
            
            # Inicializar servicios
            if self.config["local"]["enable_ml"]:
                self.services["ml_engine"] = get_ml_engine()
                await self.services["ml_engine"].start()
                logger.info("✅ Motor ML iniciado")
            
            self.services["performance_optimizer"] = get_performance_optimizer()
            await self.services["performance_optimizer"].start()
            logger.info("✅ Optimizador de rendimiento iniciado")
            
            if self.config["local"]["enable_analytics"]:
                self.services["analytics_engine"] = get_analytics_engine()
                await self.services["analytics_engine"].start()
                logger.info("✅ Motor de analytics iniciado")
            
            if self.config["local"]["enable_security_sandbox"]:
                self.services["security_sandbox"] = get_security_sandbox()
                logger.info("✅ Sandbox de seguridad iniciado")
                
        except ImportError as e:
            logger.warning(f"⚠️ Módulos avanzados no disponibles: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Error iniciando servicios avanzados: {e}")
    
    async def run_kernelhunter(self):
        """Ejecutar KernelHunter principal"""
        logger.info("🎯 Iniciando KernelHunter...")
        
        try:
            # Comando base
            cmd = ["python3", "kernelHunter.py"]
            
            # Agregar argumentos según configuración
            if self.config["local"]["enable_advanced_features"]:
                cmd.append("--advanced")
            
            # Ejecutar KernelHunter
            process = subprocess.Popen(cmd)
            self.processes.append(process)
            
            logger.info("✅ KernelHunter iniciado")
            
            # Esperar a que termine
            await asyncio.get_event_loop().run_in_executor(None, process.wait)
            
        except Exception as e:
            logger.error(f"❌ Error ejecutando KernelHunter: {e}")
    
    async def stop_services(self):
        """Detener todos los servicios"""
        logger.info("🛑 Deteniendo servicios...")
        
        # Detener procesos de Python
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        # Detener servicios avanzados
        for service_name, service in self.services.items():
            try:
                if hasattr(service, 'stop'):
                    await service.stop()
                elif hasattr(service, 'cleanup_all'):
                    service.cleanup_all()
                logger.info(f"✅ {service_name} detenido")
            except Exception as e:
                logger.warning(f"⚠️ Error deteniendo {service_name}: {e}")
        
        # Detener contenedores Docker
        docker_containers = ["prometheus", "grafana"]
        for container in docker_containers:
            try:
                subprocess.run(["docker", "stop", container], check=True)
                logger.info(f"✅ Contenedor {container} detenido")
            except:
                pass
        
        logger.info("✅ Todos los servicios detenidos")
    
    def signal_handler(self, signum, frame):
        """Manejador de señales para limpieza"""
        logger.info(f"📡 Señal {signum} recibida, limpiando...")
        asyncio.create_task(self.stop_services())
        sys.exit(0)
    
    async def run(self):
        """Ejecutar KernelHunter Advanced completo"""
        try:
            # Configurar manejador de señales
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Configurar entorno
            self.setup_environment()
            
            # Iniciar servicios
            await self.start_database_services()
            await self.start_monitoring_services()
            await self.start_web_interface()
            await self.start_advanced_services()
            
            # Mostrar información de acceso
            self.show_access_info()
            
            # Ejecutar KernelHunter
            await self.run_kernelhunter()
            
        except Exception as e:
            logger.error(f"❌ Error en ejecución: {e}")
        finally:
            await self.stop_services()
    
    def show_access_info(self):
        """Mostrar información de acceso"""
        print("\n" + "="*60)
        print("🚀 KernelHunter Advanced - Servicios Disponibles")
        print("="*60)
        
        if self.config["local"]["enable_web_interface"]:
            print(f"🌐 Dashboard Web: http://localhost:{self.config['local']['web_port']}")
        
        if self.config["monitoring"]["enable_telemetry"]:
            print(f"📊 Grafana: http://localhost:{self.config['monitoring']['grafana_port']}")
            print(f"📈 Prometheus: http://localhost:{self.config['monitoring']['prometheus_port']}")
        
        print(f"📝 Logs: logs/kernelhunter_advanced.log")
        print(f"📊 Métricas: {self.config['local']['metrics_file']}")
        print(f"💥 Crashes: {self.config['local']['crash_dir']}")
        
        print("\n🎯 Características Habilitadas:")
        if self.config["local"]["enable_ml"]:
            print("  ✅ Machine Learning")
        if self.config["local"]["enable_analytics"]:
            print("  ✅ Analytics en Tiempo Real")
        if self.config["local"]["enable_security_sandbox"]:
            print("  ✅ Sandbox de Seguridad")
        if self.config["local"]["enable_rl"]:
            print("  ✅ Reinforcement Learning")
        
        print("\n💡 Para detener: Ctrl+C")
        print("="*60 + "\n")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="KernelHunter Advanced - Ejecutor")
    parser.add_argument("--config", help="Archivo de configuración")
    parser.add_argument("--mode", choices=["local", "distributed"], default="local", help="Modo de ejecución")
    parser.add_argument("--node-id", help="ID del nodo (modo distribuido)")
    parser.add_argument("--port", type=int, help="Puerto para interfaz web")
    parser.add_argument("--generations", type=int, help="Número máximo de generaciones")
    parser.add_argument("--population", type=int, help="Tamaño de población")
    parser.add_argument("--no-ml", action="store_true", help="Deshabilitar ML")
    parser.add_argument("--no-analytics", action="store_true", help="Deshabilitar analytics")
    parser.add_argument("--no-sandbox", action="store_true", help="Deshabilitar sandbox")
    parser.add_argument("--isolation", choices=["container", "vm", "hardware"], help="Nivel de aislamiento")
    
    args = parser.parse_args()
    
    # Crear instancia
    kh = KernelHunterAdvanced(args.config)
    
    # Aplicar argumentos de línea de comandos
    if args.port:
        kh.config["local"]["web_port"] = args.port
    if args.generations:
        kh.config["local"]["max_generations"] = args.generations
    if args.population:
        kh.config["local"]["population_size"] = args.population
    if args.no_ml:
        kh.config["local"]["enable_ml"] = False
    if args.no_analytics:
        kh.config["local"]["enable_analytics"] = False
    if args.no_sandbox:
        kh.config["local"]["enable_security_sandbox"] = False
    if args.isolation:
        kh.config["local"]["sandbox_isolation_level"] = args.isolation
    if args.mode == "distributed":
        kh.config["distributed"]["enable_distributed"] = True
        if args.node_id:
            kh.config["distributed"]["node_id"] = args.node_id
    
    # Ejecutar
    asyncio.run(kh.run())

if __name__ == "__main__":
    main() 