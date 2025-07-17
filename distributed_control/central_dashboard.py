#!/usr/bin/env python3
"""
KernelHunter Distributed Control Panel
Panel central para gestionar múltiples nodos de fuzzing
"""

import asyncio
import aiohttp
import json
import time
import psycopg2
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

# Configuración
DATABASE_URL = "postgresql://user:password@localhost/kernelhunter"
API_BASE_URL = "http://localhost:8000"
NODES_CONFIG_FILE = "nodes_config.json"

@dataclass
class NodeStatus:
    node_id: str
    ip_address: str
    status: str  # 'running', 'stopped', 'error', 'crashed'
    current_generation: int
    total_crashes: int
    critical_crashes: int
    last_heartbeat: datetime
    kernel_version: str
    uptime: int
    cpu_usage: float
    memory_usage: float

@dataclass
class CrashReport:
    node_id: str
    generation: int
    crash_type: str
    shellcode_hex: str
    system_impact: bool
    timestamp: datetime
    kernel_version: str

class DistributedKernelHunter:
    def __init__(self):
        self.nodes: Dict[str, NodeStatus] = {}
        self.crash_reports: List[CrashReport] = []
        self.db_conn = None
        self.session = None
        
    async def initialize(self):
        """Inicializar conexiones y cargar configuración"""
        # Conectar a base de datos
        self.db_conn = psycopg2.connect(DATABASE_URL)
        
        # Crear sesión HTTP
        self.session = aiohttp.ClientSession()
        
        # Cargar configuración de nodos
        await self.load_nodes_config()
        
        # Crear tablas si no existen
        await self.create_database_tables()
        
    async def load_nodes_config(self):
        """Cargar configuración de nodos desde archivo"""
        try:
            with open(NODES_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                
            for node_config in config['nodes']:
                node_id = node_config['id']
                self.nodes[node_id] = NodeStatus(
                    node_id=node_id,
                    ip_address=node_config['ip'],
                    status='unknown',
                    current_generation=0,
                    total_crashes=0,
                    critical_crashes=0,
                    last_heartbeat=datetime.now(),
                    kernel_version='unknown',
                    uptime=0,
                    cpu_usage=0.0,
                    memory_usage=0.0
                )
        except FileNotFoundError:
            print(f"Config file {NODES_CONFIG_FILE} not found. Creating default config.")
            await self.create_default_config()
            
    async def create_default_config(self):
        """Crear configuración por defecto para 10 nodos"""
        config = {
            "nodes": [
                {
                    "id": f"node_{i:02d}",
                    "ip": f"192.168.1.{10+i}",
                    "ssh_port": 22,
                    "ssh_user": "root",
                    "ssh_key": "~/.ssh/id_rsa",
                    "kernel_target": "linux-5.15",
                    "max_generations": 1000,
                    "population_size": 100
                }
                for i in range(1, 11)
            ]
        }
        
        with open(NODES_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
            
    async def create_database_tables(self):
        """Crear tablas en la base de datos"""
        with self.db_conn.cursor() as cursor:
            # Tabla de nodos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id VARCHAR(50) PRIMARY KEY,
                    ip_address INET,
                    status VARCHAR(20),
                    current_generation INTEGER,
                    total_crashes INTEGER,
                    critical_crashes INTEGER,
                    last_heartbeat TIMESTAMP,
                    kernel_version VARCHAR(50),
                    uptime INTEGER,
                    cpu_usage FLOAT,
                    memory_usage FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Tabla de crashes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crashes (
                    id SERIAL PRIMARY KEY,
                    node_id VARCHAR(50),
                    generation INTEGER,
                    crash_type VARCHAR(50),
                    shellcode_hex TEXT,
                    system_impact BOOLEAN,
                    timestamp TIMESTAMP,
                    kernel_version VARCHAR(50),
                    FOREIGN KEY (node_id) REFERENCES nodes(node_id)
                )
            """)
            
            # Tabla de métricas agregadas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS aggregated_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    total_nodes INTEGER,
                    active_nodes INTEGER,
                    total_crashes INTEGER,
                    critical_crashes INTEGER,
                    avg_generation FLOAT,
                    total_system_impacts INTEGER
                )
            """)
            
        self.db_conn.commit()
        
    async def deploy_to_node(self, node_id: str):
        """Desplegar KernelHunter en un nodo específico"""
        node = self.nodes[node_id]
        
        try:
            # 1. Verificar conectividad SSH
            ssh_command = f"ssh -o ConnectTimeout=10 root@{node.ip_address} 'echo OK'"
            result = await asyncio.create_subprocess_shell(
                ssh_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await result.communicate()
            
            if result.returncode != 0:
                raise Exception(f"Cannot connect to {node.ip_address}")
                
            # 2. Instalar dependencias
            install_commands = [
                "apt-get update",
                "apt-get install -y python3 python3-pip git clang",
                "pip3 install asyncio aiohttp psycopg2-binary"
            ]
            
            for cmd in install_commands:
                ssh_cmd = f"ssh root@{node.ip_address} '{cmd}'"
                result = await asyncio.create_subprocess_shell(ssh_cmd)
                await result.communicate()
                
            # 3. Clonar KernelHunter
            clone_cmd = f"ssh root@{node.ip_address} 'cd /root && git clone https://github.com/your-repo/kernelhunter.git'"
            result = await asyncio.create_subprocess_shell(clone_cmd)
            await result.communicate()
            
            # 4. Configurar nodo específico
            config_data = {
                "node_id": node_id,
                "api_endpoint": f"http://{API_BASE_URL}/api",
                "max_generations": 1000,
                "population_size": 100,
                "use_rl_weights": True
            }
            
            config_json = json.dumps(config_data)
            config_cmd = f"ssh root@{node.ip_address} 'echo \'{config_json}\' > /root/kernelhunter/node_config.json'"
            result = await asyncio.create_subprocess_shell(config_cmd)
            await result.communicate()
            
            # 5. Iniciar KernelHunter
            start_cmd = f"ssh root@{node.ip_address} 'cd /root/kernelhunter && nohup python3 kernelHunter.py --node-id {node_id} --api-mode > kernelhunter.log 2>&1 &'"
            result = await asyncio.create_subprocess_shell(start_cmd)
            await result.communicate()
            
            # 6. Actualizar estado
            node.status = 'running'
            await self.update_node_status(node_id)
            
            print(f"✅ Node {node_id} deployed successfully")
            
        except Exception as e:
            print(f"❌ Failed to deploy node {node_id}: {e}")
            node.status = 'error'
            await self.update_node_status(node_id)
            
    async def deploy_all_nodes(self):
        """Desplegar KernelHunter en todos los nodos"""
        print("🚀 Deploying KernelHunter to all nodes...")
        
        # Desplegar en paralelo
        tasks = [self.deploy_to_node(node_id) for node_id in self.nodes.keys()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        print("✅ Deployment completed")
        
    async def monitor_nodes(self):
        """Monitorear estado de todos los nodos"""
        while True:
            print(f"\n📊 Monitoring {len(self.nodes)} nodes at {datetime.now()}")
            
            for node_id in self.nodes.keys():
                await self.check_node_health(node_id)
                
            # Actualizar métricas agregadas
            await self.update_aggregated_metrics()
            
            # Esperar 30 segundos antes del siguiente check
            await asyncio.sleep(30)
            
    async def check_node_health(self, node_id: str):
        """Verificar salud de un nodo específico"""
        node = self.nodes[node_id]
        
        try:
            # Verificar API del nodo
            api_url = f"http://{node.ip_address}:8000/api/status"
            async with self.session.get(api_url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Actualizar estado del nodo
                    node.status = data.get('status', 'unknown')
                    node.current_generation = data.get('current_generation', 0)
                    node.total_crashes = data.get('total_crashes', 0)
                    node.critical_crashes = data.get('critical_crashes', 0)
                    node.last_heartbeat = datetime.now()
                    node.kernel_version = data.get('kernel_version', 'unknown')
                    node.uptime = data.get('uptime', 0)
                    node.cpu_usage = data.get('cpu_usage', 0.0)
                    node.memory_usage = data.get('memory_usage', 0.0)
                    
                    # Sincronizar crashes
                    await self.sync_crashes_from_node(node_id)
                    
                else:
                    node.status = 'error'
                    
        except Exception as e:
            print(f"❌ Node {node_id} health check failed: {e}")
            node.status = 'error'
            
        # Actualizar en base de datos
        await self.update_node_status(node_id)
        
    async def sync_crashes_from_node(self, node_id: str):
        """Sincronizar crashes desde un nodo"""
        node = self.nodes[node_id]
        
        try:
            # Obtener crashes recientes
            api_url = f"http://{node.ip_address}:8000/api/crashes/recent"
            async with self.session.get(api_url, timeout=10) as response:
                if response.status == 200:
                    crashes_data = await response.json()
                    
                    for crash_data in crashes_data.get('crashes', []):
                        crash = CrashReport(
                            node_id=node_id,
                            generation=crash_data['generation'],
                            crash_type=crash_data['crash_type'],
                            shellcode_hex=crash_data['shellcode_hex'],
                            system_impact=crash_data['system_impact'],
                            timestamp=datetime.fromisoformat(crash_data['timestamp']),
                            kernel_version=node.kernel_version
                        )
                        
                        # Verificar si ya existe
                        if not await self.crash_exists(crash):
                            await self.save_crash(crash)
                            
        except Exception as e:
            print(f"❌ Failed to sync crashes from node {node_id}: {e}")
            
    async def crash_exists(self, crash: CrashReport) -> bool:
        """Verificar si un crash ya existe en la base de datos"""
        with self.db_conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM crashes 
                WHERE node_id = %s AND generation = %s AND shellcode_hex = %s
            """, (crash.node_id, crash.generation, crash.shellcode_hex))
            
            count = cursor.fetchone()[0]
            return count > 0
            
    async def save_crash(self, crash: CrashReport):
        """Guardar crash en la base de datos"""
        with self.db_conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO crashes (node_id, generation, crash_type, shellcode_hex, 
                                   system_impact, timestamp, kernel_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (crash.node_id, crash.generation, crash.crash_type, crash.shellcode_hex,
                  crash.system_impact, crash.timestamp, crash.kernel_version))
            
        self.db_conn.commit()
        print(f"💥 New crash from {crash.node_id}: {crash.crash_type}")
        
    async def update_node_status(self, node_id: str):
        """Actualizar estado de nodo en la base de datos"""
        node = self.nodes[node_id]
        
        with self.db_conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO nodes (node_id, ip_address, status, current_generation,
                                 total_crashes, critical_crashes, last_heartbeat,
                                 kernel_version, uptime, cpu_usage, memory_usage)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (node_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    current_generation = EXCLUDED.current_generation,
                    total_crashes = EXCLUDED.total_crashes,
                    critical_crashes = EXCLUDED.critical_crashes,
                    last_heartbeat = EXCLUDED.last_heartbeat,
                    kernel_version = EXCLUDED.kernel_version,
                    uptime = EXCLUDED.uptime,
                    cpu_usage = EXCLUDED.cpu_usage,
                    memory_usage = EXCLUDED.memory_usage
            """, (node.node_id, node.ip_address, node.status, node.current_generation,
                  node.total_crashes, node.critical_crashes, node.last_heartbeat,
                  node.kernel_version, node.uptime, node.cpu_usage, node.memory_usage))
            
        self.db_conn.commit()
        
    async def update_aggregated_metrics(self):
        """Actualizar métricas agregadas"""
        active_nodes = sum(1 for node in self.nodes.values() if node.status == 'running')
        total_crashes = sum(node.total_crashes for node in self.nodes.values())
        critical_crashes = sum(node.critical_crashes for node in self.nodes.values())
        avg_generation = sum(node.current_generation for node in self.nodes.values()) / max(1, len(self.nodes))
        
        with self.db_conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO aggregated_metrics 
                (timestamp, total_nodes, active_nodes, total_crashes, 
                 critical_crashes, avg_generation, total_system_impacts)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (datetime.now(), len(self.nodes), active_nodes, total_crashes,
                  critical_crashes, avg_generation, critical_crashes))
            
        self.db_conn.commit()
        
    async def get_dashboard_data(self) -> Dict:
        """Obtener datos para el dashboard"""
        # Estadísticas de nodos
        node_stats = {
            'total': len(self.nodes),
            'running': sum(1 for n in self.nodes.values() if n.status == 'running'),
            'stopped': sum(1 for n in self.nodes.values() if n.status == 'stopped'),
            'error': sum(1 for n in self.nodes.values() if n.status == 'error')
        }
        
        # Estadísticas de crashes
        total_crashes = sum(n.total_crashes for n in self.nodes.values())
        critical_crashes = sum(n.critical_crashes for n in self.nodes.values())
        
        # Nodos con más crashes
        top_crash_nodes = sorted(
            self.nodes.values(), 
            key=lambda x: x.total_crashes, 
            reverse=True
        )[:5]
        
        return {
            'node_stats': node_stats,
            'total_crashes': total_crashes,
            'critical_crashes': critical_crashes,
            'top_crash_nodes': [
                {
                    'node_id': n.node_id,
                    'crashes': n.total_crashes,
                    'critical': n.critical_crashes,
                    'generation': n.current_generation
                }
                for n in top_crash_nodes
            ],
            'nodes': [
                {
                    'id': n.node_id,
                    'status': n.status,
                    'generation': n.current_generation,
                    'crashes': n.total_crashes,
                    'last_heartbeat': n.last_heartbeat.isoformat()
                }
                for n in self.nodes.values()
            ]
        }
        
    async def stop_node(self, node_id: str):
        """Detener KernelHunter en un nodo"""
        node = self.nodes[node_id]
        
        try:
            stop_cmd = f"ssh root@{node.ip_address} 'pkill -f kernelHunter.py'"
            result = await asyncio.create_subprocess_shell(stop_cmd)
            await result.communicate()
            
            node.status = 'stopped'
            await self.update_node_status(node_id)
            
            print(f"🛑 Node {node_id} stopped")
            
        except Exception as e:
            print(f"❌ Failed to stop node {node_id}: {e}")
            
    async def stop_all_nodes(self):
        """Detener todos los nodos"""
        print("🛑 Stopping all nodes...")
        
        tasks = [self.stop_node(node_id) for node_id in self.nodes.keys()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        print("✅ All nodes stopped")
        
    async def cleanup(self):
        """Limpiar recursos"""
        if self.session:
            await self.session.close()
        if self.db_conn:
            self.db_conn.close()

async def main():
    """Función principal"""
    controller = DistributedKernelHunter()
    
    try:
        await controller.initialize()
        
        # Desplegar en todos los nodos
        await controller.deploy_all_nodes()
        
        # Iniciar monitoreo
        await controller.monitor_nodes()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        await controller.stop_all_nodes()
    finally:
        await controller.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 