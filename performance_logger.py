#!/usr/bin/env python3
"""
Sistema de logging detallado para KernelHunter que registra métricas de rendimiento,
permitiendo comparar la efectividad con y sin reinforcement learning.
Guarda información detallada de cada generación, crashes, y métricas de diversidad.
"""

import json
import csv
import os
import time
from datetime import datetime
from collections import Counter
from typing import Dict, List, Any, Optional

# Archivos de log
PERFORMANCE_LOG_FILE = "kernelhunter_performance.json"
PERFORMANCE_CSV_FILE = "kernelhunter_performance.csv"
CRASH_DETAIL_LOG = "kernelhunter_crash_detail.json"
RL_COMPARISON_LOG = "kernelhunter_rl_comparison.json"

class PerformanceLogger:
    def __init__(self, session_name: str = None):
        """
        Inicializa el logger de rendimiento.
        
        Args:
            session_name: Nombre de la sesión de fuzzing (opcional)
        """
        self.session_name = session_name or f"session_{int(time.time())}"
        self.session_start = time.time()
        self.log_data = {
            "session_info": {
                "session_name": self.session_name,
                "start_time": datetime.now().isoformat(),
                "start_timestamp": self.session_start,
                "use_rl_weights": False,
                "total_runtime": 0,
                "generations_completed": 0
            },
            "generations": [],
            "crashes": [],
            "rl_metrics": [],
            "diversity_evolution": [],
            "performance_summary": {}
        }
        
        # Inicializar archivo CSV si no existe
        self._init_csv_file()
        
    def set_rl_mode(self, use_rl: bool):
        """Establece si esta sesión usa reinforcement learning."""
        self.log_data["session_info"]["use_rl_weights"] = use_rl
        
    def log_generation(self, 
                      generation_id: int,
                      population_size: int,
                      crash_rate: float,
                      system_impacts: int,
                      avg_shellcode_length: float,
                      crash_types: Dict[str, int],
                      attack_stats: Dict[str, int] = None,
                      mutation_stats: Dict[str, int] = None,
                      rl_weights: Dict[str, List[float]] = None,
                      diversity_metrics: Dict[str, float] = None):
        """
        Registra información detallada de una generación.
        
        Args:
            generation_id: ID de la generación
            population_size: Tamaño de la población
            crash_rate: Tasa de crashes (0.0 - 1.0)
            system_impacts: Número de impactos al sistema
            avg_shellcode_length: Longitud promedio del shellcode
            crash_types: Diccionario con tipos de crashes y sus conteos
            attack_stats: Estadísticas de ataques usados
            mutation_stats: Estadísticas de mutaciones usadas
            rl_weights: Pesos de RL si están disponibles
            diversity_metrics: Métricas de diversidad genética
        """
        generation_data = {
            "generation_id": generation_id,
            "timestamp": datetime.now().isoformat(),
            "unix_timestamp": time.time(),
            "runtime_seconds": time.time() - self.session_start,
            "population_size": population_size,
            "crash_rate": crash_rate,
            "system_impacts": system_impacts,
            "avg_shellcode_length": avg_shellcode_length,
            "crash_types": crash_types,
            "total_crashes": sum(crash_types.values()) if crash_types else 0,
            "unique_crash_types": len(crash_types) if crash_types else 0,
            "attack_stats": attack_stats or {},
            "mutation_stats": mutation_stats or {},
            "rl_weights": rl_weights or {},
            "diversity_metrics": diversity_metrics or {}
        }
        
        self.log_data["generations"].append(generation_data)
        self.log_data["session_info"]["generations_completed"] = generation_id + 1
        
        # Escribir a CSV para análisis rápido
        self._write_csv_row(generation_data)
        
        # Guardar archivo JSON completo cada 10 generaciones
        if generation_id % 10 == 0:
            self._save_json_log()
            
    def log_crash_detail(self,
                        generation_id: int,
                        program_id: int,
                        shellcode_hex: str,
                        crash_type: str,
                        return_code: int,
                        system_impact: bool,
                        parent_shellcode_hex: str = None,
                        attack_type: str = None,
                        mutation_type: str = None):
        """
        Registra detalles específicos de un crash.
        
        Args:
            generation_id: ID de la generación
            program_id: ID del programa que crasheó
            shellcode_hex: Shellcode en hexadecimal
            crash_type: Tipo de crash (SIGSEGV, SIGILL, etc.)
            return_code: Código de retorno
            system_impact: Si tuvo impacto a nivel de sistema
            parent_shellcode_hex: Shellcode padre (opcional)
            attack_type: Tipo de ataque usado (opcional)
            mutation_type: Tipo de mutación usado (opcional)
        """
        crash_data = {
            "generation_id": generation_id,
            "program_id": program_id,
            "timestamp": datetime.now().isoformat(),
            "shellcode_hex": shellcode_hex,
            "shellcode_length": len(bytes.fromhex(shellcode_hex)) if shellcode_hex else 0,
            "crash_type": crash_type,
            "return_code": return_code,
            "system_impact": system_impact,
            "parent_shellcode_hex": parent_shellcode_hex,
            "attack_type": attack_type,
            "mutation_type": mutation_type
        }
        
        self.log_data["crashes"].append(crash_data)
        
    def log_rl_metrics(self,
                      generation_id: int,
                      attack_q_values: List[float],
                      mutation_q_values: List[float],
                      attack_counts: List[int],
                      mutation_counts: List[int],
                      epsilon: float,
                      recent_rewards: Dict[str, float] = None):
        """
        Registra métricas específicas del reinforcement learning.
        
        Args:
            generation_id: ID de la generación
            attack_q_values: Valores Q para ataques
            mutation_q_values: Valores Q para mutaciones
            attack_counts: Conteos de ataques
            mutation_counts: Conteos de mutaciones
            epsilon: Valor actual de epsilon
            recent_rewards: Recompensas recientes (opcional)
        """
        rl_data = {
            "generation_id": generation_id,
            "timestamp": datetime.now().isoformat(),
            "attack_q_values": attack_q_values,
            "mutation_q_values": mutation_q_values,
            "attack_counts": attack_counts,
            "mutation_counts": mutation_counts,
            "epsilon": epsilon,
            "recent_rewards": recent_rewards or {}
        }
        
        self.log_data["rl_metrics"].append(rl_data)
        
    def log_diversity_metrics(self,
                             generation_id: int,
                             reservoir_size: int,
                             avg_diversity: float,
                             unique_instruction_types: int,
                             length_variance: float,
                             genetic_health: str):
        """
        Registra métricas de diversidad genética.
        
        Args:
            generation_id: ID de la generación
            reservoir_size: Tamaño del reservoir genético
            avg_diversity: Diversidad promedio
            unique_instruction_types: Tipos únicos de instrucciones
            length_variance: Varianza en la longitud de shellcodes
            genetic_health: Estado de salud genética (string)
        """
        diversity_data = {
            "generation_id": generation_id,
            "timestamp": datetime.now().isoformat(),
            "reservoir_size": reservoir_size,
            "avg_diversity": avg_diversity,
            "unique_instruction_types": unique_instruction_types,
            "length_variance": length_variance,
            "genetic_health": genetic_health
        }
        
        self.log_data["diversity_evolution"].append(diversity_data)
        
    def finalize_session(self):
        """
        Finaliza la sesión y calcula métricas de resumen.
        """
        end_time = time.time()
        total_runtime = end_time - self.session_start
        
        self.log_data["session_info"]["end_time"] = datetime.now().isoformat()
        self.log_data["session_info"]["total_runtime"] = total_runtime
        
        # Calcular métricas de resumen
        generations = self.log_data["generations"]
        crashes = self.log_data["crashes"]
        
        if generations:
            summary = {
                "total_generations": len(generations),
                "avg_crash_rate": sum(g["crash_rate"] for g in generations) / len(generations),
                "total_system_impacts": sum(g["system_impacts"] for g in generations),
                "avg_shellcode_length": sum(g["avg_shellcode_length"] for g in generations) / len(generations),
                "total_crashes": len(crashes),
                "unique_crash_types": len(set(c["crash_type"] for c in crashes)),
                "system_impact_rate": sum(1 for c in crashes if c["system_impact"]) / len(crashes) if crashes else 0,
                "crashes_per_minute": len(crashes) / (total_runtime / 60) if total_runtime > 0 else 0,
                "generations_per_hour": len(generations) / (total_runtime / 3600) if total_runtime > 0 else 0
            }
            
            # Análisis de tendencias
            if len(generations) >= 2:
                first_half = generations[:len(generations)//2]
                second_half = generations[len(generations)//2:]
                
                summary["trend_analysis"] = {
                    "crash_rate_improvement": (
                        sum(g["crash_rate"] for g in second_half) / len(second_half) -
                        sum(g["crash_rate"] for g in first_half) / len(first_half)
                    ),
                    "system_impact_trend": (
                        sum(g["system_impacts"] for g in second_half) / len(second_half) -
                        sum(g["system_impacts"] for g in first_half) / len(first_half)
                    )
                }
            
            self.log_data["performance_summary"] = summary
        
        # Guardar log final
        self._save_json_log()
        self._save_crash_detail_log()
        
    def _init_csv_file(self):
        """Inicializa el archivo CSV con headers."""
        if not os.path.exists(PERFORMANCE_CSV_FILE):
            headers = [
                "session_name", "generation_id", "timestamp", "runtime_seconds",
                "use_rl", "population_size", "crash_rate", "system_impacts",
                "avg_shellcode_length", "total_crashes", "unique_crash_types",
                "diversity_avg", "epsilon"
            ]
            
            with open(PERFORMANCE_CSV_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
    def _write_csv_row(self, generation_data):
        """Escribe una fila al archivo CSV."""
        row = [
            self.session_name,
            generation_data["generation_id"],
            generation_data["timestamp"],
            generation_data["runtime_seconds"],
            self.log_data["session_info"]["use_rl_weights"],
            generation_data["population_size"],
            generation_data["crash_rate"],
            generation_data["system_impacts"],
            generation_data["avg_shellcode_length"],
            generation_data["total_crashes"],
            generation_data["unique_crash_types"],
            generation_data["diversity_metrics"].get("diversity_avg", 0),
            generation_data["rl_weights"].get("epsilon", 0) if "rl_weights" in generation_data else 0
        ]
        
        with open(PERFORMANCE_CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
    def _save_json_log(self):
        """Guarda el log completo en formato JSON."""
        with open(PERFORMANCE_LOG_FILE, 'w') as f:
            json.dump(self.log_data, f, indent=2)
            
    def _save_crash_detail_log(self):
        """Guarda log detallado de crashes."""
        crash_log = {
            "session_name": self.session_name,
            "crashes": self.log_data["crashes"],
            "summary": {
                "total_crashes": len(self.log_data["crashes"]),
                "crash_types": Counter(c["crash_type"] for c in self.log_data["crashes"]),
                "system_impacts": sum(1 for c in self.log_data["crashes"] if c["system_impact"])
            }
        }
        
        with open(CRASH_DETAIL_LOG, 'w') as f:
            json.dump(crash_log, f, indent=2)

# Funciones de análisis y comparación
def compare_rl_vs_no_rl(rl_session_file: str, no_rl_session_file: str):
    """
    Compara dos sesiones: una con RL y otra sin RL.
    
    Args:
        rl_session_file: Archivo de la sesión con RL
        no_rl_session_file: Archivo de la sesión sin RL
        
    Returns:
        Dict con métricas comparativas
    """
    def load_session(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    rl_data = load_session(rl_session_file)
    no_rl_data = load_session(no_rl_session_file)
    
    if not rl_data or not no_rl_data:
        return {"error": "Could not load session files"}
    
    comparison = {
        "sessions_compared": {
            "rl_session": rl_data["session_info"]["session_name"],
            "no_rl_session": no_rl_data["session_info"]["session_name"]
        },
        "performance_comparison": {},
        "crash_analysis": {},
        "efficiency_metrics": {}
    }
    
    # Comparar rendimiento general
    rl_summary = rl_data.get("performance_summary", {})
    no_rl_summary = no_rl_data.get("performance_summary", {})
    
    if rl_summary and no_rl_summary:
        comparison["performance_comparison"] = {
            "crash_rate_improvement": rl_summary.get("avg_crash_rate", 0) - no_rl_summary.get("avg_crash_rate", 0),
            "system_impacts_diff": rl_summary.get("total_system_impacts", 0) - no_rl_summary.get("total_system_impacts", 0),
            "crashes_per_minute_diff": rl_summary.get("crashes_per_minute", 0) - no_rl_summary.get("crashes_per_minute", 0),
            "rl_efficiency": rl_summary.get("crashes_per_minute", 0) / no_rl_summary.get("crashes_per_minute", 1) if no_rl_summary.get("crashes_per_minute", 0) > 0 else 0
        }
    
    # Análisis detallado de crashes
    rl_crashes = Counter(c["crash_type"] for c in rl_data.get("crashes", []))
    no_rl_crashes = Counter(c["crash_type"] for c in no_rl_data.get("crashes", []))
    
    comparison["crash_analysis"] = {
        "rl_unique_crash_types": len(rl_crashes),
        "no_rl_unique_crash_types": len(no_rl_crashes),
        "rl_only_crashes": list(set(rl_crashes.keys()) - set(no_rl_crashes.keys())),
        "no_rl_only_crashes": list(set(no_rl_crashes.keys()) - set(rl_crashes.keys())),
        "common_crashes": list(set(rl_crashes.keys()) & set(no_rl_crashes.keys()))
    }
    
    return comparison

def generate_performance_report(session_file: str = None):
    """
    Genera un reporte de rendimiento detallado.
    
    Args:
        session_file: Archivo de sesión específico (opcional)
        
    Returns:
        String con el reporte formateado
    """
    if session_file and os.path.exists(session_file):
        with open(session_file, 'r') as f:
            data = json.load(f)
    elif os.path.exists(PERFORMANCE_LOG_FILE):
        with open(PERFORMANCE_LOG_FILE, 'r') as f:
            data = json.load(f)
    else:
        return "No performance data available"
    
    session_info = data["session_info"]
    summary = data.get("performance_summary", {})
    
    report = f"""
KERNELHUNTER PERFORMANCE REPORT
===============================

Session: {session_info.get('session_name', 'Unknown')}
Start Time: {session_info.get('start_time', 'Unknown')}
Runtime: {summary.get('total_runtime', 0) / 3600:.2f} hours
RL Enabled: {session_info.get('use_rl_weights', False)}

PERFORMANCE METRICS
-------------------
Total Generations: {summary.get('total_generations', 0)}
Average Crash Rate: {summary.get('avg_crash_rate', 0):.2%}
Total System Impacts: {summary.get('total_system_impacts', 0)}
Total Crashes: {summary.get('total_crashes', 0)}
Unique Crash Types: {summary.get('unique_crash_types', 0)}
System Impact Rate: {summary.get('system_impact_rate', 0):.2%}

EFFICIENCY METRICS
------------------
Crashes per Minute: {summary.get('crashes_per_minute', 0):.2f}
Generations per Hour: {summary.get('generations_per_hour', 0):.2f}
Avg Shellcode Length: {summary.get('avg_shellcode_length', 0):.1f} bytes
"""
    
    # Añadir análisis de tendencias si está disponible
    trend = summary.get("trend_analysis", {})
    if trend:
        report += f"""
TREND ANALYSIS
--------------
Crash Rate Change: {trend.get('crash_rate_improvement', 0):+.3f}
System Impact Trend: {trend.get('system_impact_trend', 0):+.1f}
"""
    
    return report

# Ejemplo de uso
if __name__ == "__main__":
    # Crear logger para sesión de prueba
    logger = PerformanceLogger("test_session_rl")
    logger.set_rl_mode(True)
    
    # Simular algunas generaciones
    for gen in range(5):
        logger.log_generation(
            generation_id=gen,
            population_size=100,
            crash_rate=0.7 + random.uniform(-0.1, 0.1),
            system_impacts=random.randint(2, 8),
            avg_shellcode_length=25.5 + random.uniform(-5, 5),
            crash_types={"SIGSEGV": 15, "SIGILL": 5, "SIGTRAP": 2},
            attack_stats={"memory_access": 20, "syscall": 15},
            mutation_stats={"add": 10, "modify": 8},
            rl_weights={"epsilon": 0.1, "learning_rate": 0.01},
            diversity_metrics={"diversity_avg": 0.6}
        )
    
    # Finalizar sesión
    logger.finalize_session()
    
    print("Performance logging test completed!")
    print(generate_performance_report())
