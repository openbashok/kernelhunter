#!/usr/bin/env python3
# genetic_reservoir.py - VERSIÓN PROFESIONAL FINAL
# Sistema de diversidad genética profesional para KernelHunter
# 100% compatible y optimizado

import random
import numpy as np
from collections import Counter
import math

# === IMPORTS PROFESIONALES CON FALLBACKS ===
try:
    import textdistance
    TEXTDISTANCE_AVAILABLE = True
except ImportError:
    TEXTDISTANCE_AVAILABLE = False

try:
    from Bio import Align
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

try:
    from scipy.spatial.distance import hamming
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class GeneticReservoir:
    """
    Implements an advanced genetic reservoir with professional diversity algorithms.
    Maintains a diverse population of shellcodes to ensure genetic variety
    and prevent premature convergence or stagnation in evolution.
    """
    
    def __init__(self, max_size=100, diversity_threshold=0.7):
        """
        Initializes the genetic reservoir with professional diversity system.
        
        Args:
            max_size: Maximum size of the reservoir
            diversity_threshold: Minimum diversity threshold to include new individuals
        """
        self.reservoir = []
        self.max_size = max_size
        self.diversity_threshold = diversity_threshold
        self.crash_types = set()  # Registry of observed crash types
        self.features_cache = {}  # Cache of extracted features
        
        # === SISTEMA PROFESIONAL ===
        self._setup_professional_diversity()
        
        # Performance metrics
        self._diversity_calculations = 0
        self._last_optimization = 0
        self._performance_history = []
        
    def _setup_professional_diversity(self):
        """Configura el sistema de diversidad profesional"""
        self.use_enhanced_diversity = False
        self.available_algorithms = []
        
        # Configurar textdistance (más eficiente)
        if TEXTDISTANCE_AVAILABLE:
            try:
                self.textdistance_levenshtein = textdistance.Levenshtein()
                self.textdistance_jaccard = textdistance.Jaccard(qval=2)
                # Test rápido
                test_result = self.textdistance_levenshtein.normalized_distance("test", "best")
                if 0 <= test_result <= 1:
                    self.use_enhanced_diversity = True
                    self.available_algorithms.append("textdistance")
            except Exception:
                pass
        
        # Configurar BioPython (para análisis de secuencias)
        if BIOPYTHON_AVAILABLE:
            try:
                self.biopython_aligner = Align.PairwiseAligner()
                self.biopython_aligner.mode = 'local'
                self.biopython_aligner.match_score = 2
                self.biopython_aligner.mismatch_score = -1
                self.biopython_aligner.open_gap_score = -2
                self.biopython_aligner.extend_gap_score = -0.5
                self.available_algorithms.append("biopython")
            except Exception:
                self.biopython_aligner = None
        
        # Configurar SciPy (fallback)
        if SCIPY_AVAILABLE:
            self.available_algorithms.append("scipy")
    
    def __len__(self):
        """Returns the number of shellcodes in the reservoir."""
        return len(self.reservoir)
        
    def calculate_diversity(self, shellcode1, shellcode2):
        """
        Calcula diversidad usando sistema profesional multi-algoritmo.
        Preserva interfaz original pero con algoritmos avanzados.
        
        Args:
            shellcode1: First shellcode (bytes)
            shellcode2: Second shellcode (bytes)
            
        Returns:
            float: Value between 0 (identical) and 1 (completely different)
        """
        self._diversity_calculations += 1
        
        # Auto-optimización inteligente cada 500 cálculos
        if self._diversity_calculations % 500 == 0:
            self._intelligent_auto_optimization()
        
        # Usar sistema profesional si está disponible
        if self.use_enhanced_diversity:
            try:
                return self._calculate_diversity_professional(shellcode1, shellcode2)
            except Exception:
                # Fallback silencioso
                self.use_enhanced_diversity = False
        
        # Fallback al método original mejorado
        return self._calculate_diversity_original(shellcode1, shellcode2)
    
    def _calculate_diversity_professional(self, shellcode1, shellcode2):
        """
        Sistema de diversidad profesional que combina múltiples algoritmos
        """
        if len(shellcode1) == 0 or len(shellcode2) == 0:
            return 1.0
        
        # Estrategia adaptativa según tamaño
        max_len = max(len(shellcode1), len(shellcode2))
        
        if max_len > 10000:  # Shellcodes muy grandes
            return self._calculate_diversity_large_shellcodes(shellcode1, shellcode2)
        elif max_len > 1000:  # Shellcodes grandes
            return self._calculate_diversity_medium_shellcodes(shellcode1, shellcode2)
        else:  # Shellcodes pequeños
            return self._calculate_diversity_small_shellcodes(shellcode1, shellcode2)
    
    def _calculate_diversity_large_shellcodes(self, shellcode1, shellcode2):
        """Optimizado para shellcodes muy grandes (>10KB)"""
        # Usar muestreo inteligente para eficiencia
        sample1 = self._get_representative_sample(shellcode1, 2000)
        sample2 = self._get_representative_sample(shellcode2, 2000)
        
        # Diversidad estructural (muestreada)
        str1 = sample1.hex()
        str2 = sample2.hex()
        
        if "textdistance" in self.available_algorithms:
            structural_div = self.textdistance_levenshtein.normalized_distance(str1, str2)
        else:
            structural_div = self._hamming_distance_normalized(sample1, sample2)
        
        # Diversidad de características globales
        feature_div = self._calculate_feature_diversity(shellcode1, shellcode2)
        
        # Diversidad de entropía
        entropy_div = self._calculate_entropy_diversity(shellcode1, shellcode2)
        
        # Combinar con pesos optimizados para shellcodes grandes
        return 0.4 * structural_div + 0.4 * feature_div + 0.2 * entropy_div
    
    def _calculate_diversity_medium_shellcodes(self, shellcode1, shellcode2):
        """Optimizado para shellcodes medianos (1-10KB)"""
        # Usar algoritmos completos pero optimizados
        
        if "textdistance" in self.available_algorithms:
            # Usar Jaccard para n-gramas (más eficiente que Levenshtein completo)
            str1 = shellcode1.hex()
            str2 = shellcode2.hex()
            structural_div = self.textdistance_jaccard.normalized_distance(str1, str2)
        elif "biopython" in self.available_algorithms:
            structural_div = self._biopython_diversity(shellcode1, shellcode2)
        else:
            structural_div = self._hamming_distance_normalized(shellcode1, shellcode2)
        
        # Diversidad de frecuencias de bytes
        freq_div = self._calculate_byte_frequency_diversity(shellcode1, shellcode2)
        
        # Diversidad de características
        feature_div = self._calculate_feature_diversity(shellcode1, shellcode2)
        
        # Combinar con pesos balanceados
        return 0.5 * structural_div + 0.3 * freq_div + 0.2 * feature_div
    
    def _calculate_diversity_small_shellcodes(self, shellcode1, shellcode2):
        """Optimizado para shellcodes pequeños (<1KB)"""
        # Usar análisis completo y detallado
        
        if "textdistance" in self.available_algorithms:
            str1 = shellcode1.hex()
            str2 = shellcode2.hex()
            
            # Usar múltiples algoritmos para mayor precisión
            levenshtein_div = self.textdistance_levenshtein.normalized_distance(str1, str2)
            jaccard_div = self.textdistance_jaccard.normalized_distance(str1, str2)
            structural_div = 0.6 * levenshtein_div + 0.4 * jaccard_div
        else:
            structural_div = self._hamming_distance_normalized(shellcode1, shellcode2)
        
        # Análisis detallado de instrucciones
        feature_div = self._calculate_feature_diversity(shellcode1, shellcode2)
        
        # Diversidad posicional (importante en shellcodes pequeños)
        positional_div = self._calculate_positional_diversity(shellcode1, shellcode2)
        
        # Combinar con pesos que priorizan precisión
        return 0.5 * structural_div + 0.3 * feature_div + 0.2 * positional_div
    
    def _get_representative_sample(self, shellcode, target_size):
        """Obtiene muestra representativa de shellcode grande"""
        if len(shellcode) <= target_size:
            return shellcode
        
        # Tomar inicio, medio y final
        chunk_size = target_size // 3
        start = shellcode[:chunk_size]
        middle_pos = len(shellcode) // 2 - chunk_size // 2
        middle = shellcode[middle_pos:middle_pos + chunk_size]
        end = shellcode[-chunk_size:]
        
        return start + middle + end
    
    def _calculate_feature_diversity(self, shellcode1, shellcode2):
        """Diversidad basada en características extraídas"""
        id1 = self._get_shellcode_id(shellcode1)
        id2 = self._get_shellcode_id(shellcode2)
        
        # Extraer características si no están en caché
        if id1 not in self.features_cache:
            self.extract_features(shellcode1)
        if id2 not in self.features_cache:
            self.extract_features(shellcode2)
        
        feat1 = self.features_cache[id1]
        feat2 = self.features_cache[id2]
        
        # Diversidad de tipos de instrucciones (método original mejorado)
        instruction_div = self._calculate_instruction_diversity(
            feat1["instruction_types"], 
            feat2["instruction_types"]
        )
        
        # Diferencia de longitud normalizada
        length_diff = abs(feat1["length"] - feat2["length"]) / max(feat1["length"], feat2["length"])
        
        # Diferencia de syscalls
        syscall_diff = abs(feat1["syscalls"] - feat2["syscalls"]) / (max(feat1["syscalls"], feat2["syscalls"]) + 1)
        
        # Diferencia de instrucciones privilegiadas
        priv_diff = abs(feat1["privileged_instr"] - feat2["privileged_instr"]) / (max(feat1["privileged_instr"], feat2["privileged_instr"]) + 1)
        
        # Combinar con pesos optimizados
        return 0.4 * instruction_div + 0.25 * length_diff + 0.2 * syscall_diff + 0.15 * priv_diff
    
    def _calculate_entropy_diversity(self, shellcode1, shellcode2):
        """Diversidad basada en entropía de Shannon"""
        entropy1 = self._calculate_entropy(shellcode1)
        entropy2 = self._calculate_entropy(shellcode2)
        
        max_entropy = max(entropy1, entropy2, 0.001)
        return abs(entropy1 - entropy2) / max_entropy
    
    def _calculate_entropy(self, data):
        """Calcula entropía de Shannon"""
        if not data:
            return 0.0
        
        freq = Counter(data)
        length = len(data)
        
        entropy = 0.0
        for count in freq.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_byte_frequency_diversity(self, shellcode1, shellcode2):
        """Diversidad de distribución de frecuencias de bytes"""
        freq1 = Counter(shellcode1)
        freq2 = Counter(shellcode2)
        
        # Normalizar frecuencias
        total1 = len(shellcode1) if shellcode1 else 1
        total2 = len(shellcode2) if shellcode2 else 1
        
        norm_freq1 = {b: count/total1 for b, count in freq1.items()}
        norm_freq2 = {b: count/total2 for b, count in freq2.items()}
        
        # Calcular divergencia de Jensen-Shannon simplificada
        all_bytes = set(norm_freq1.keys()) | set(norm_freq2.keys())
        
        js_divergence = 0.0
        for byte_val in all_bytes:
            p = norm_freq1.get(byte_val, 0)
            q = norm_freq2.get(byte_val, 0)
            m = (p + q) / 2
            
            if p > 0 and m > 0:
                js_divergence += p * math.log2(p / m)
            if q > 0 and m > 0:
                js_divergence += q * math.log2(q / m)
        
        return min(1.0, js_divergence / 2.0)
    
    def _calculate_positional_diversity(self, shellcode1, shellcode2):
        """Diversidad posicional para shellcodes pequeños"""
        max_len = max(len(shellcode1), len(shellcode2))
        
        if max_len == 0:
            return 0.0
        
        # Padding
        padded1 = shellcode1 + bytes([0] * (max_len - len(shellcode1)))
        padded2 = shellcode2 + bytes([0] * (max_len - len(shellcode2)))
        
        # Contar diferencias posicionales
        differences = sum(1 for a, b in zip(padded1, padded2) if a != b)
        
        return differences / max_len
    
    def _hamming_distance_normalized(self, shellcode1, shellcode2):
        """Distancia de Hamming normalizada"""
        max_len = max(len(shellcode1), len(shellcode2))
        
        if max_len == 0:
            return 0.0
        
        # Padding con bytes nulos
        padded1 = list(shellcode1) + [0] * (max_len - len(shellcode1))
        padded2 = list(shellcode2) + [0] * (max_len - len(shellcode2))
        
        if "scipy" in self.available_algorithms:
            try:
                return hamming(padded1, padded2)
            except Exception:
                pass
        
        # Implementación manual
        differences = sum(1 for a, b in zip(padded1, padded2) if a != b)
        return differences / max_len
    
    def _biopython_diversity(self, shellcode1, shellcode2):
        """Diversidad usando BioPython Smith-Waterman"""
        if not hasattr(self, 'biopython_aligner') or self.biopython_aligner is None:
            return self._hamming_distance_normalized(shellcode1, shellcode2)
        
        # Convertir a secuencias hexadecimales
        seq1 = ''.join(f'{b:02x}' for b in shellcode1)
        seq2 = ''.join(f'{b:02x}' for b in shellcode2)
        
        try:
            score = self.biopython_aligner.score(seq1, seq2)
            max_possible = min(len(seq1), len(seq2)) * self.biopython_aligner.match_score
            
            if max_possible > 0:
                similarity = score / max_possible
                return 1.0 - max(0.0, min(1.0, similarity))
            else:
                return 1.0
        except Exception:
            return self._hamming_distance_normalized(shellcode1, shellcode2)
    
    def _intelligent_auto_optimization(self):
        """Auto-optimización inteligente basada en métricas de rendimiento"""
        if len(self.reservoir) < 10:
            return
        
        try:
            # Calcular diversidad promedio de una muestra
            sample_size = min(10, len(self.reservoir))
            diversities = []
            
            for i in range(sample_size):
                for j in range(i+1, sample_size):
                    div = self._calculate_diversity_original(self.reservoir[i], self.reservoir[j])
                    diversities.append(div)
            
            if diversities:
                avg_diversity = sum(diversities) / len(diversities)
                self._performance_history.append(avg_diversity)
                
                # Mantener solo últimas 10 métricas
                if len(self._performance_history) > 10:
                    self._performance_history = self._performance_history[-10:]
                
                # Detectar tendencias
                if len(self._performance_history) >= 5:
                    recent_avg = sum(self._performance_history[-3:]) / 3
                    
                    # Ajustes inteligentes
                    if recent_avg < 0.2:  # Diversidad muy baja
                        self.diversity_threshold = max(0.3, self.diversity_threshold - 0.1)
                    elif recent_avg > 0.9 and len(self.reservoir) < self.max_size * 0.7:  # Diversidad muy alta, reservorio poco lleno
                        self.diversity_threshold = min(0.8, self.diversity_threshold + 0.05)
                    
        except Exception:
            pass  # Fallar silenciosamente
    
    # === MÉTODOS ORIGINALES PRESERVADOS ===
    def _calculate_diversity_original(self, shellcode1, shellcode2):
        """Método original preservado exactamente"""
        if len(shellcode1) == 0 or len(shellcode2) == 0:
            return 1.0
            
        id1 = self._get_shellcode_id(shellcode1)
        id2 = self._get_shellcode_id(shellcode2)
        
        if id1 in self.features_cache and id2 in self.features_cache:
            feat1 = self.features_cache[id1]
            feat2 = self.features_cache[id2]
            
            instruction_div = self._calculate_instruction_diversity(
                feat1["instruction_types"], 
                feat2["instruction_types"]
            )
            
            length_diff = abs(feat1["length"] - feat2["length"]) / max(feat1["length"], feat2["length"])
            syscall_diff = abs(feat1["syscalls"] - feat2["syscalls"]) / (max(feat1["syscalls"], feat2["syscalls"]) + 1)
            
            return 0.4 * instruction_div + 0.3 * length_diff + 0.3 * syscall_diff
        
        # Fallback method
        edit_distance = sum(a != b for a, b in zip(shellcode1[:min(len(shellcode1), len(shellcode2))], 
                                                 shellcode2[:min(len(shellcode1), len(shellcode2))]))
        length_diff = abs(len(shellcode1) - len(shellcode2))
        max_len = max(len(shellcode1), len(shellcode2))
        normalized_distance = (edit_distance + length_diff) / max_len if max_len > 0 else 1.0
        
        return normalized_distance
    
    def _calculate_instruction_diversity(self, types1, types2):
        """Método original preservado exactamente"""
        all_keys = set(types1.keys()) | set(types2.keys())
        
        weights = {
            "known_vulns": 2.5, "privileged": 2.0, "control_registers": 2.0,
            "speculative_exec": 2.0, "syscall": 1.5, "memory_access": 1.5,
            "segment_registers": 1.5, "forced_exception": 1.5, "control_flow": 1.2,
            "stack_manipulation": 1.2, "arithmetic": 1.0, "simd": 1.0,
            "x86_opcode": 1.0, "other": 0.5
        }
        
        weighted_diff = 0
        weighted_total = 0
        
        for key in all_keys:
            val1 = types1.get(key, 0)
            val2 = types2.get(key, 0)
            key_weight = weights.get(key, 1.0)
            
            weighted_diff += abs(val1 - val2) * key_weight
            weighted_total += max(val1, val2) * key_weight
        
        return weighted_diff / weighted_total if weighted_total > 0 else 1.0
    
    def _get_shellcode_id(self, shellcode):
        """Método original preservado exactamente"""
        prefix = shellcode[:min(10, len(shellcode))].hex()
        suffix = shellcode[-min(10, len(shellcode)):].hex()
        length = len(shellcode)
        return f"{prefix}_{length}_{suffix}"
    
    def extract_features(self, shellcode):
        """Método original preservado exactamente"""
        shellcode_id = self._get_shellcode_id(shellcode)
        if shellcode_id in self.features_cache:
            return self.features_cache[shellcode_id]
        
        features = {
            "length": len(shellcode),
            "syscalls": self._count_syscalls(shellcode),
            "privileged_instr": self._count_privileged_instructions(shellcode),
            "instruction_types": self._analyze_instruction_types(shellcode),
        }
        
        self.features_cache[shellcode_id] = features
        return features
    
    def is_diverse_enough(self, shellcode):
        """Método original preservado exactamente"""
        if not self.reservoir:
            return True
            
        self.extract_features(shellcode)
        diversities = [self.calculate_diversity(shellcode, existing) 
                      for existing in self.reservoir]
        
        return min(diversities) > self.diversity_threshold
    
    def add(self, shellcode, crash_info=None):
        """Método original con guardado automático optimizado"""
        # Check duplicates
        for existing in self.reservoir:
            if shellcode == existing:
                print(f"Shellcode rejected: Duplicate")
                return False
        
        # Add logic
        if len(self.reservoir) < self.max_size:
            if self.is_diverse_enough(shellcode):
                self.reservoir.append(shellcode)
                print(f"Added shellcode to reservoir (size now: {len(self.reservoir)})")
                if crash_info:
                    self.crash_types.add(crash_info.get("crash_type", "unknown"))
                
                # Guardado automático inteligente
                if len(self.reservoir) % 20 == 0:  # Cada 20 en lugar de 10
                    self._auto_save()
                
                return True
            else:
                print(f"Shellcode rejected: Not diverse enough")
        else:
            if self.is_diverse_enough(shellcode):
                diversity_scores = []
                for i, sc in enumerate(self.reservoir):
                    avg_diversity = sum(self.calculate_diversity(sc, other) 
                                       for j, other in enumerate(self.reservoir) if i != j) / (len(self.reservoir) - 1)
                    diversity_scores.append((i, avg_diversity))
                
                least_diverse_idx = min(diversity_scores, key=lambda x: x[1])[0]
                self.reservoir[least_diverse_idx] = shellcode
                
                if crash_info:
                    self.crash_types.add(crash_info.get("crash_type", "unknown"))
                
                # Guardado automático
                if len(self.reservoir) % 20 == 0:
                    self._auto_save()
                
                return True
        
        return False
    
    def _auto_save(self):
        """Guardado automático optimizado"""
        try:
            self.save_to_file("kernelhunter_reservoir.pkl")
        except Exception:
            pass  # Fallar silenciosamente
    
    def get_sample(self, n=1):
        """Método original preservado exactamente"""
        if not self.reservoir:
            return []
        return random.sample(self.reservoir, min(n, len(self.reservoir)))
    
    def get_diverse_sample(self, n=1):
        """Método original optimizado para mejor rendimiento"""
        if len(self.reservoir) <= n:
            return self.reservoir.copy()
            
        # Optimización: para n grande, usar muestreo más eficiente
        if n > len(self.reservoir) // 2:
            # Si pedimos más de la mitad, es más eficiente usar todos menos algunos
            excluded_count = len(self.reservoir) - n
            excluded_indices = set(random.sample(range(len(self.reservoir)), excluded_count))
            return [sc for i, sc in enumerate(self.reservoir) if i not in excluded_indices]
        
        # Método original para n pequeño
        selected = [random.choice(self.reservoir)]
        
        while len(selected) < n:
            max_diversities = []
            
            for candidate in self.reservoir:
                if candidate in selected:
                    continue
                    
                min_div = min(self.calculate_diversity(candidate, s) for s in selected)
                max_diversities.append((candidate, min_div))
            
            if max_diversities:
                next_selection = max(max_diversities, key=lambda x: x[1])[0]
                selected.append(next_selection)
            else:
                break
                
        return selected
    
    def get_by_feature(self, feature_name, value, comparison="gt", limit=5):
        """Método original preservado exactamente"""
        results = []
        
        for shellcode in self.reservoir:
            shellcode_id = self._get_shellcode_id(shellcode)
            if shellcode_id not in self.features_cache:
                self.extract_features(shellcode)
                
            features = self.features_cache[shellcode_id]
            
            if feature_name in features:
                feature_value = features[feature_name]
                
                if comparison == "gt" and feature_value > value:
                    results.append(shellcode)
                elif comparison == "lt" and feature_value < value:
                    results.append(shellcode)
                elif comparison == "eq" and feature_value == value:
                    results.append(shellcode)
                    
                if len(results) >= limit:
                    break
                    
        return results
    
    def save_to_file(self, filename):
        """Método original con metadatos profesionales"""
        import pickle
        
        data = {
            "reservoir": self.reservoir,
            "crash_types": self.crash_types,
            "max_size": self.max_size,
            "diversity_threshold": self.diversity_threshold,
            # Metadatos profesionales
            "_professional_metadata": {
                "version": "2.0",
                "diversity_calculations": self._diversity_calculations,
                "available_algorithms": self.available_algorithms,
                "use_enhanced_diversity": self.use_enhanced_diversity,
                "performance_history": self._performance_history[-5:]  # Solo últimas 5
            }
        }
        
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    
    def load_from_file(self, filename):
        """Método original con carga de metadatos profesionales"""
        import pickle
        
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                
            self.reservoir = data["reservoir"]
            self.crash_types = data["crash_types"]
            self.max_size = data["max_size"]
            self.diversity_threshold = data["diversity_threshold"]
            
            # Cargar metadatos profesionales si existen
            if "_professional_metadata" in data:
                metadata = data["_professional_metadata"]
                self._diversity_calculations = metadata.get("diversity_calculations", 0)
                self._performance_history = metadata.get("performance_history", [])
            
            # Recalcular cache de características
            self.features_cache = {}
            for shellcode in self.reservoir:
                self.extract_features(shellcode)
                
            return True
            
        except (FileNotFoundError, KeyError, pickle.PickleError):
            return False
    
    def _count_syscalls(self, shellcode):
        """Método original preservado exactamente"""
        syscall_pattern = b"\x0f\x05"
        int80_pattern = b"\xcd\x80"
        
        count = shellcode.count(syscall_pattern)
        count += shellcode.count(int80_pattern)
        
        syscall_setup = b"\x48\xc7\xc0"
        setup_count = shellcode.count(syscall_setup)
        
        if setup_count > count:
            count += (setup_count - count) // 2
            
        return count
    
    def _count_privileged_instructions(self, shellcode):
        """Método original preservado exactamente"""
        privileged_patterns = [
            b"\x0f\x01", b"\xf4", b"\x0f\x00", b"\x0f\x06", b"\x0f\x09",
            b"\x0f\x30", b"\x0f\x32", b"\x0f\x34", b"\x0f\x35", b"\x0f\x07",
            b"\x0f\x22", b"\x0f\x20", b"\x0f\x01\xd0", b"\x0f\x01\xf8",
        ]
        
        count = 0
        for pattern in privileged_patterns:
            count += shellcode.count(pattern)
            
        return count
    
    def _analyze_instruction_types(self, shellcode):
        """Método original optimizado para mejor rendimiento"""
        types = {
            "syscall": 0, "memory_access": 0, "privileged": 0, "control_flow": 0,
            "arithmetic": 0, "simd": 0, "segment_registers": 0, "speculative_exec": 0,
            "forced_exception": 0, "control_registers": 0, "stack_manipulation": 0,
            "known_vulns": 0, "x86_opcode": 0, "other": 0
        }
        
        # Optimización: para shellcodes muy grandes, usar muestreo
        if len(shellcode) > 50000:
            sample = self._get_representative_sample(shellcode, 10000)
            return self._analyze_instruction_types_detailed(sample)
        elif len(shellcode) > 10000:
            sample = self._get_representative_sample(shellcode, 5000)
            return self._analyze_instruction_types_detailed(sample)
        else:
            return self._analyze_instruction_types_detailed(shellcode)
    
    def _analyze_instruction_types_detailed(self, shellcode):
        """Análisis detallado de tipos de instrucciones"""
        types = {
            "syscall": 0, "memory_access": 0, "privileged": 0, "control_flow": 0,
            "arithmetic": 0, "simd": 0, "segment_registers": 0, "speculative_exec": 0,
            "forced_exception": 0, "control_registers": 0, "stack_manipulation": 0,
            "known_vulns": 0, "x86_opcode": 0, "other": 0
        }
        
        # Patrones optimizados (solo los más importantes para eficiencia)
        patterns = {
            "syscall": [b"\x0f\x05", b"\xcd\x80"],
            "memory_access": [b"\x48\x8b", b"\x48\x89", b"\xff", b"\x0f\xae"],
            "privileged": [b"\x0f\x01", b"\xf4", b"\x0f\x30", b"\x0f\x32"],
            "control_flow": [b"\xe9", b"\xeb", b"\x74", b"\x75", b"\xe8", b"\xc3"],
            "arithmetic": [b"\x48\x01", b"\x48\x29", b"\x48\xf7"],
            "simd": [b"\x0f\x10", b"\x0f\x11", b"\x0f\x28", b"\x0f\x29"],
            "forced_exception": [b"\x0f\x0b", b"\xcc", b"\xcd\x03"],
            "known_vulns": [b"\x0f\xae\x05", b"\x65\x48\x8b\x04\x25", b"\x0f\x3f"]
        }
        
        # Análisis optimizado
        i = 0
        while i < len(shellcode):
            matched = False
            
            # Buscar patrones importantes primero
            for category in ["known_vulns", "privileged", "syscall", "memory_access"]:
                if category in patterns:
                    for pattern in patterns[category]:
                        if i <= len(shellcode) - len(pattern) and shellcode[i:i+len(pattern)] == pattern:
                            types[category] += 1
                            i += len(pattern)
                            matched = True
                            break
                if matched:
                    break
            
            if not matched:
                # Buscar otros patrones
                for category, pattern_list in patterns.items():
                    if category in ["known_vulns", "privileged", "syscall", "memory_access"]:
                        continue  # Ya los revisamos
                    
                    for pattern in pattern_list:
                        if i <= len(shellcode) - len(pattern) and shellcode[i:i+len(pattern)] == pattern:
                            types[category] += 1
                            i += len(pattern)
                            matched = True
                            break
                    if matched:
                        break
            
            if not matched:
                # Verificar prefijo REX
                if i < len(shellcode) and 0x40 <= shellcode[i] <= 0x4F:
                    types["x86_opcode"] += 1
                    i += min(4, len(shellcode) - i)  # Saltar instrucción estimada
                else:
                    types["other"] += 1
                    i += 1
        
        return types
    
    def get_diversity_stats(self):
        """Estadísticas de diversidad con métricas profesionales"""
        if len(self.reservoir) < 2:
            return {
                "diversity_avg": 0, "diversity_min": 0, "diversity_max": 0,
                "professional_enabled": self.use_enhanced_diversity,
                "available_algorithms": self.available_algorithms,
                "total_calculations": self._diversity_calculations
            }
            
        # Calcular diversidad (muestra limitada para eficiencia)
        sample_size = min(20, len(self.reservoir))  # Limitar para eficiencia
        diversities = []
        
        indices = random.sample(range(len(self.reservoir)), sample_size)
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                div = self.calculate_diversity(self.reservoir[indices[i]], self.reservoir[indices[j]])
                diversities.append(div)
        
        # Contar tipos de instrucciones
        instruction_types_counts = Counter()
        for i in indices:  # Solo muestra para eficiencia
            shellcode = self.reservoir[i]
            shellcode_id = self._get_shellcode_id(shellcode)
            if shellcode_id not in self.features_cache:
                self.extract_features(shellcode)
            
            types = self.features_cache[shellcode_id]["instruction_types"]
            for type_name, count in types.items():
                if count > 0:
                    instruction_types_counts[type_name] += 1
                
        return {
            "diversity_avg": sum(diversities) / len(diversities) if diversities else 0,
            "diversity_min": min(diversities) if diversities else 0,
            "diversity_max": max(diversities) if diversities else 0,
            "unique_crash_types": len(self.crash_types),
            "reservoir_size": len(self.reservoir),
            "avg_shellcode_length": sum(len(sc) for sc in self.reservoir) / len(self.reservoir),
            "instruction_types_distribution": dict(instruction_types_counts.most_common()),
            # Métricas profesionales
            "professional_enabled": self.use_enhanced_diversity,
            "available_algorithms": self.available_algorithms,
            "total_calculations": self._diversity_calculations,
            "current_threshold": self.diversity_threshold,
            "performance_trend": self._performance_history[-3:] if len(self._performance_history) >= 3 else []
        }
    
    def clear_cache(self):
        """Limpieza optimizada de caché"""
        # Mantener caché de shellcodes más usados
        if len(self.features_cache) > 1000:
            # Mantener solo los últimos 500
            cache_items = list(self.features_cache.items())
            self.features_cache = dict(cache_items[-500:])
        else:
            self.features_cache = {}
    
    # === MÉTODOS ADICIONALES PROFESIONALES ===
    def get_professional_status(self):
        """Obtiene estado completo del sistema profesional"""
        return {
            "version": "2.0",
            "professional_enabled": self.use_enhanced_diversity,
            "available_algorithms": self.available_algorithms,
            "primary_algorithm": self.available_algorithms[0] if self.available_algorithms else "original",
            "total_calculations": self._diversity_calculations,
            "reservoir_utilization": len(self.reservoir) / self.max_size,
            "current_threshold": self.diversity_threshold,
            "cache_size": len(self.features_cache),
            "performance_history": self._performance_history,
            "auto_optimization_active": True
        }
    
    def benchmark_diversity_performance(self, test_shellcodes=None):
        """Benchmark de rendimiento de algoritmos de diversidad"""
        import time
        
        if test_shellcodes is None:
            # Usar shellcodes del reservorio si no se proporcionan
            if len(self.reservoir) >= 2:
                test_shellcodes = self.reservoir[:2]
            else:
                # Shellcodes de prueba
                test_shellcodes = [
                    b"\x48\xc7\xc0\x3c\x00\x00\x00\x48\x31\xff\x0f\x05",  # exit
                    b"\x48\xc7\xc0\x01\x00\x00\x00\x48\x31\xff\x0f\x05"   # write
                ]
        
        results = {}
        
        # Benchmark algoritmo original
        start_time = time.time()
        original_result = self._calculate_diversity_original(test_shellcodes[0], test_shellcodes[1])
        original_time = time.time() - start_time
        results["original"] = {"result": original_result, "time_ms": original_time * 1000}
        
        # Benchmark algoritmo profesional
        if self.use_enhanced_diversity:
            start_time = time.time()
            professional_result = self._calculate_diversity_professional(test_shellcodes[0], test_shellcodes[1])
            professional_time = time.time() - start_time
            results["professional"] = {"result": professional_result, "time_ms": professional_time * 1000}
            
            # Calcular speedup/slowdown
            if original_time > 0:
                results["performance_ratio"] = professional_time / original_time
                results["speedup"] = f"{results['performance_ratio']:.2f}x"
        
        results["algorithm_used"] = self.available_algorithms[0] if self.available_algorithms else "original"
        return results
    
    def optimize_for_large_shellcodes(self):
        """Optimización específica para shellcodes grandes"""
        large_shellcode_count = sum(1 for sc in self.reservoir if len(sc) > 10000)
        
        if large_shellcode_count > len(self.reservoir) * 0.3:  # Más del 30% son grandes
            # Ajustar estrategia para shellcodes grandes
            self.diversity_threshold = max(0.4, self.diversity_threshold - 0.1)
            print(f"🔧 Optimized for large shellcodes: threshold adjusted to {self.diversity_threshold:.2f}")
    
    def get_size_distribution(self):
        """Obtiene distribución de tamaños de shellcodes"""
        if not self.reservoir:
            return {}
        
        sizes = [len(sc) for sc in self.reservoir]
        sizes.sort()
        
        return {
            "min_size": min(sizes),
            "max_size": max(sizes),
            "median_size": sizes[len(sizes)//2],
            "avg_size": sum(sizes) / len(sizes),
            "small_count": sum(1 for s in sizes if s < 1000),
            "medium_count": sum(1 for s in sizes if 1000 <= s < 10000),
            "large_count": sum(1 for s in sizes if s >= 10000),
            "size_distribution": {
                "< 1KB": sum(1 for s in sizes if s < 1000),
                "1-10KB": sum(1 for s in sizes if 1000 <= s < 10000),
                "10-100KB": sum(1 for s in sizes if 10000 <= s < 100000),
                "> 100KB": sum(1 for s in sizes if s >= 100000)
            }
        }


# === INICIALIZACIÓN PROFESIONAL ===
def _initialize_professional_system():
    """Inicialización del sistema profesional"""
    libs = []
    if TEXTDISTANCE_AVAILABLE:
        libs.append("textdistance")
    if BIOPYTHON_AVAILABLE:
        libs.append("biopython")
    if SCIPY_AVAILABLE:
        libs.append("scipy")
    
    if libs:
        print(f"🧬 GeneticReservoir Professional v2.0: Multi-algorithm diversity system ({', '.join(libs)})")
    else:
        print("🧬 GeneticReservoir v2.0: Enhanced original algorithm")

# Inicializar automáticamente
_initialize_professional_system()


# === INSTRUCCIONES DE USO ===
"""
SISTEMA PROFESIONAL DE DIVERSIDAD GENÉTICA v2.0
===============================================

CARACTERÍSTICAS:
✅ Multi-algoritmo adaptativo (TextDistance + BioPython + SciPy)
✅ Optimización automática de parámetros
✅ Análisis diferenciado por tamaño de shellcode
✅ Auto-guardado inteligente cada 20 adiciones
✅ Caché optimizado para eficiencia
✅ Métricas de rendimiento avanzadas
✅ 100% compatible con código original

ALGORITMOS INCLUIDOS:
- Levenshtein normalizada (secuencias pequeñas)
- Jaccard con n-gramas (secuencias medianas)
- Muestreo representativo (secuencias grandes >10KB)
- Smith-Waterman local (BioPython)
- Diversidad de entropía (Shannon)
- Divergencia Jensen-Shannon (frecuencias)

OPTIMIZACIONES:
- Shellcodes >100KB: Muestreo inteligente
- Shellcodes 1-10KB: Algoritmos completos optimizados
- Shellcodes <1KB: Análisis detallado multi-algoritmo
- Auto-ajuste de threshold basado en tendencias
- Guardado automático cada 20 vs 10 adiciones

INSTALACIÓN:
1. Reemplazar genetic_reservoir.py con este código
2. Opcional: pip install textdistance biopython scipy
3. Ejecutar kernelhunter.py normalmente

MÉTODOS ADICIONALES:
- reservoir.get_professional_status()
- reservoir.benchmark_diversity_performance()
- reservoir.optimize_for_large_shellcodes()
- reservoir.get_size_distribution()

COMPATIBILIDAD: 100% con kernelhunter.py existente
"""
