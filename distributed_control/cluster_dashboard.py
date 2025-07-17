#!/usr/bin/env python3
"""
KernelHunter Distributed Cluster Dashboard
Dashboard web para monitorear múltiples nodos de fuzzing
"""

import asyncio
import aiohttp
from aiohttp import web
import json
import psycopg2
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import os

# Configuración
DATABASE_URL = "postgresql://user:password@localhost/kernelhunter"
PORT = 8080
HOST = "0.0.0.0"

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterDashboard:
    def __init__(self):
        self.db_conn = None
        self.app = web.Application()
        self.setup_routes()
        
    def setup_routes(self):
        """Configurar rutas del dashboard"""
        self.app.router.add_get('/', self.dashboard)
        self.app.router.add_get('/api/nodes', self.api_nodes)
        self.app.router.add_get('/api/crashes', self.api_crashes)
        self.app.router.add_get('/api/metrics', self.api_metrics)
        self.app.router.add_get('/api/node/{node_id}', self.api_node_detail)
        self.app.router.add_post('/api/node/{node_id}/restart', self.api_restart_node)
        self.app.router.add_static('/static', 'static')
        
    async def connect_db(self):
        """Conectar a la base de datos"""
        try:
            self.db_conn = psycopg2.connect(DATABASE_URL)
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
            
    async def dashboard(self, request):
        """Página principal del dashboard"""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>KernelHunter Distributed Cluster</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                }
                .header h1 {
                    font-size: 2.5em;
                    margin: 0;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .stat-card {
                    background: rgba(255,255,255,0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 20px;
                    text-align: center;
                    border: 1px solid rgba(255,255,255,0.2);
                }
                .stat-card h3 {
                    margin: 0 0 10px 0;
                    font-size: 1.2em;
                    opacity: 0.9;
                }
                .stat-card .value {
                    font-size: 2.5em;
                    font-weight: bold;
                    margin: 10px 0;
                }
                .nodes-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .node-card {
                    background: rgba(255,255,255,0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 20px;
                    border: 1px solid rgba(255,255,255,0.2);
                }
                .node-card h3 {
                    margin: 0 0 15px 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .status-badge {
                    padding: 5px 10px;
                    border-radius: 20px;
                    font-size: 0.8em;
                    font-weight: bold;
                }
                .status-running { background: #4CAF50; }
                .status-stopped { background: #f44336; }
                .status-error { background: #ff9800; }
                .node-metrics {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 10px;
                    margin-top: 15px;
                }
                .metric {
                    text-align: center;
                }
                .metric .label {
                    font-size: 0.8em;
                    opacity: 0.8;
                }
                .metric .value {
                    font-size: 1.2em;
                    font-weight: bold;
                }
                .charts-container {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .chart-card {
                    background: rgba(255,255,255,0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 20px;
                    border: 1px solid rgba(255,255,255,0.2);
                }
                .chart-card h3 {
                    margin: 0 0 15px 0;
                    text-align: center;
                }
                .refresh-btn {
                    background: rgba(255,255,255,0.2);
                    border: none;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 25px;
                    cursor: pointer;
                    font-size: 1em;
                    margin-bottom: 20px;
                }
                .refresh-btn:hover {
                    background: rgba(255,255,255,0.3);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 KernelHunter Distributed Cluster</h1>
                    <p>Monitoreo en tiempo real de 10 nodos de fuzzing</p>
                </div>
                
                <button class="refresh-btn" onclick="refreshData()">🔄 Actualizar Datos</button>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Nodos Activos</h3>
                        <div class="value" id="active-nodes">-</div>
                    </div>
                    <div class="stat-card">
                        <h3>Total Crashes</h3>
                        <div class="value" id="total-crashes">-</div>
                    </div>
                    <div class="stat-card">
                        <h3>Crashes Críticos</h3>
                        <div class="value" id="critical-crashes">-</div>
                    </div>
                    <div class="stat-card">
                        <h3>Generación Promedio</h3>
                        <div class="value" id="avg-generation">-</div>
                    </div>
                </div>
                
                <div class="charts-container">
                    <div class="chart-card">
                        <h3>Crashes por Nodo</h3>
                        <canvas id="crashesChart"></canvas>
                    </div>
                    <div class="chart-card">
                        <h3>Evolución de Crashes</h3>
                        <canvas id="evolutionChart"></canvas>
                    </div>
                </div>
                
                <div class="nodes-grid" id="nodes-container">
                    <!-- Los nodos se cargarán dinámicamente -->
                </div>
            </div>
            
            <script>
                let crashesChart, evolutionChart;
                
                async function refreshData() {
                    try {
                        // Cargar datos de nodos
                        const nodesResponse = await fetch('/api/nodes');
                        const nodesData = await nodesResponse.json();
                        
                        // Cargar métricas
                        const metricsResponse = await fetch('/api/metrics');
                        const metricsData = await metricsResponse.json();
                        
                        // Actualizar estadísticas
                        document.getElementById('active-nodes').textContent = metricsData.active_nodes;
                        document.getElementById('total-crashes').textContent = metricsData.total_crashes;
                        document.getElementById('critical-crashes').textContent = metricsData.critical_crashes;
                        document.getElementById('avg-generation').textContent = metricsData.avg_generation.toFixed(1);
                        
                        // Actualizar nodos
                        updateNodesGrid(nodesData.nodes);
                        
                        // Actualizar gráficos
                        updateCharts(nodesData, metricsData);
                        
                    } catch (error) {
                        console.error('Error refreshing data:', error);
                    }
                }
                
                function updateNodesGrid(nodes) {
                    const container = document.getElementById('nodes-container');
                    container.innerHTML = '';
                    
                    nodes.forEach(node => {
                        const nodeCard = document.createElement('div');
                        nodeCard.className = 'node-card';
                        nodeCard.innerHTML = `
                            <h3>
                                ${node.id}
                                <span class="status-badge status-${node.status}">${node.status}</span>
                            </h3>
                            <div class="node-metrics">
                                <div class="metric">
                                    <div class="label">Generación</div>
                                    <div class="value">${node.current_generation}</div>
                                </div>
                                <div class="metric">
                                    <div class="label">Crashes</div>
                                    <div class="value">${node.total_crashes}</div>
                                </div>
                                <div class="metric">
                                    <div class="label">Críticos</div>
                                    <div class="value">${node.critical_crashes}</div>
                                </div>
                                <div class="metric">
                                    <div class="label">Último HB</div>
                                    <div class="value">${formatTime(node.last_heartbeat)}</div>
                                </div>
                            </div>
                        `;
                        container.appendChild(nodeCard);
                    });
                }
                
                function updateCharts(nodesData, metricsData) {
                    // Gráfico de crashes por nodo
                    const crashesCtx = document.getElementById('crashesChart').getContext('2d');
                    if (crashesChart) crashesChart.destroy();
                    
                    crashesChart = new Chart(crashesCtx, {
                        type: 'bar',
                        data: {
                            labels: nodesData.nodes.map(n => n.id),
                            datasets: [{
                                label: 'Total Crashes',
                                data: nodesData.nodes.map(n => n.total_crashes),
                                backgroundColor: 'rgba(255, 99, 132, 0.8)',
                                borderColor: 'rgba(255, 99, 132, 1)',
                                borderWidth: 1
                            }, {
                                label: 'Critical Crashes',
                                data: nodesData.nodes.map(n => n.critical_crashes),
                                backgroundColor: 'rgba(255, 159, 64, 0.8)',
                                borderColor: 'rgba(255, 159, 64, 1)',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                    
                    // Gráfico de evolución
                    const evolutionCtx = document.getElementById('evolutionChart').getContext('2d');
                    if (evolutionChart) evolutionChart.destroy();
                    
                    evolutionChart = new Chart(evolutionCtx, {
                        type: 'line',
                        data: {
                            labels: metricsData.evolution_labels || [],
                            datasets: [{
                                label: 'Total Crashes',
                                data: metricsData.evolution_crashes || [],
                                borderColor: 'rgba(255, 99, 132, 1)',
                                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                }
                
                function formatTime(timestamp) {
                    const date = new Date(timestamp);
                    const now = new Date();
                    const diff = now - date;
                    
                    if (diff < 60000) return 'Ahora';
                    if (diff < 3600000) return Math.floor(diff / 60000) + 'm';
                    return Math.floor(diff / 3600000) + 'h';
                }
                
                // Cargar datos iniciales
                refreshData();
                
                // Actualizar cada 30 segundos
                setInterval(refreshData, 30000);
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
        
    async def api_nodes(self, request):
        """API para obtener estado de nodos"""
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT node_id, ip_address, status, current_generation,
                           total_crashes, critical_crashes, last_heartbeat,
                           kernel_version, uptime, cpu_usage, memory_usage
                    FROM nodes
                    ORDER BY node_id
                """)
                
                nodes = []
                for row in cursor.fetchall():
                    nodes.append({
                        'id': row[0],
                        'ip': row[1],
                        'status': row[2],
                        'current_generation': row[3],
                        'total_crashes': row[4],
                        'critical_crashes': row[5],
                        'last_heartbeat': row[6].isoformat() if row[6] else None,
                        'kernel_version': row[7],
                        'uptime': row[8],
                        'cpu_usage': row[9],
                        'memory_usage': row[10]
                    })
                
                return web.json_response({'nodes': nodes})
                
        except Exception as e:
            logger.error(f"Error getting nodes: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def api_crashes(self, request):
        """API para obtener crashes recientes"""
        try:
            limit = int(request.query.get('limit', 50))
            
            with self.db_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT node_id, generation, crash_type, shellcode_hex,
                           system_impact, timestamp, kernel_version
                    FROM crashes
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (limit,))
                
                crashes = []
                for row in cursor.fetchall():
                    crashes.append({
                        'node_id': row[0],
                        'generation': row[1],
                        'crash_type': row[2],
                        'shellcode_hex': row[3][:32] + '...' if len(row[3]) > 32 else row[3],
                        'system_impact': row[4],
                        'timestamp': row[5].isoformat() if row[5] else None,
                        'kernel_version': row[6]
                    })
                
                return web.json_response({'crashes': crashes})
                
        except Exception as e:
            logger.error(f"Error getting crashes: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def api_metrics(self, request):
        """API para obtener métricas agregadas"""
        try:
            with self.db_conn.cursor() as cursor:
                # Métricas actuales
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_nodes,
                        COUNT(CASE WHEN status = 'running' THEN 1 END) as active_nodes,
                        SUM(total_crashes) as total_crashes,
                        SUM(critical_crashes) as critical_crashes,
                        AVG(current_generation) as avg_generation
                    FROM nodes
                """)
                
                row = cursor.fetchone()
                metrics = {
                    'total_nodes': row[0] or 0,
                    'active_nodes': row[1] or 0,
                    'total_crashes': row[2] or 0,
                    'critical_crashes': row[3] or 0,
                    'avg_generation': float(row[4] or 0)
                }
                
                # Evolución temporal (últimas 24 horas)
                cursor.execute("""
                    SELECT 
                        DATE_TRUNC('hour', timestamp) as hour,
                        COUNT(*) as crashes
                    FROM crashes
                    WHERE timestamp >= NOW() - INTERVAL '24 hours'
                    GROUP BY hour
                    ORDER BY hour
                """)
                
                evolution_data = cursor.fetchall()
                metrics['evolution_labels'] = [row[0].strftime('%H:%M') for row in evolution_data]
                metrics['evolution_crashes'] = [row[1] for row in evolution_data]
                
                return web.json_response(metrics)
                
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def api_node_detail(self, request):
        """API para obtener detalles de un nodo específico"""
        node_id = request.match_info['node_id']
        
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute("""
                    SELECT node_id, ip_address, status, current_generation,
                           total_crashes, critical_crashes, last_heartbeat,
                           kernel_version, uptime, cpu_usage, memory_usage
                    FROM nodes
                    WHERE node_id = %s
                """, (node_id,))
                
                row = cursor.fetchone()
                if not row:
                    return web.json_response({'error': 'Node not found'}, status=404)
                
                node_data = {
                    'id': row[0],
                    'ip': row[1],
                    'status': row[2],
                    'current_generation': row[3],
                    'total_crashes': row[4],
                    'critical_crashes': row[5],
                    'last_heartbeat': row[6].isoformat() if row[6] else None,
                    'kernel_version': row[7],
                    'uptime': row[8],
                    'cpu_usage': row[9],
                    'memory_usage': row[10]
                }
                
                # Obtener crashes recientes del nodo
                cursor.execute("""
                    SELECT generation, crash_type, system_impact, timestamp
                    FROM crashes
                    WHERE node_id = %s
                    ORDER BY timestamp DESC
                    LIMIT 10
                """, (node_id,))
                
                recent_crashes = []
                for crash_row in cursor.fetchall():
                    recent_crashes.append({
                        'generation': crash_row[0],
                        'crash_type': crash_row[1],
                        'system_impact': crash_row[2],
                        'timestamp': crash_row[3].isoformat() if crash_row[3] else None
                    })
                
                node_data['recent_crashes'] = recent_crashes
                
                return web.json_response(node_data)
                
        except Exception as e:
            logger.error(f"Error getting node detail: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def api_restart_node(self, request):
        """API para reiniciar un nodo"""
        node_id = request.match_info['node_id']
        
        try:
            # Aquí implementarías la lógica para reiniciar el nodo
            # Por ejemplo, enviar comando SSH o usar la API de Linode
            
            logger.info(f"Restarting node {node_id}")
            
            return web.json_response({'message': f'Node {node_id} restart initiated'})
            
        except Exception as e:
            logger.error(f"Error restarting node: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def start(self):
        """Iniciar el dashboard"""
        await self.connect_db()
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, HOST, PORT)
        await site.start()
        
        logger.info(f"Dashboard started at http://{HOST}:{PORT}")
        
        # Mantener el servidor corriendo
        try:
            await asyncio.Future()  # Esperar indefinidamente
        except KeyboardInterrupt:
            logger.info("Shutting down dashboard...")
        finally:
            await runner.cleanup()
            if self.db_conn:
                self.db_conn.close()

async def main():
    """Función principal"""
    dashboard = ClusterDashboard()
    await dashboard.start()

if __name__ == "__main__":
    asyncio.run(main()) 