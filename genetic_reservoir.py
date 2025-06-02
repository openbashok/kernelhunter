#!/usr/bin/env python3
# genetic_reservoir.py - VERSIÓN TRANSPARENTE COMPLETA
# 100% compatible con kernelhunter.py sin modificaciones
# Reemplaza completamente el archivo genetic_reservoir.py original

import random
import numpy as np
from collections import Counter
import hashlib
import math

# === IMPORTS PROFESIONALES CON FALLBACKS SILENCIOSOS ===
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
    from scipy.spatial.distance import hamming, jaccard
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class GeneticReservoir:
    """
    Implementa un reservorio genético mejorado con diversidad profesional.
    COMPLETAMENTE TRANSPARENTE - No requiere cambios en kernelhunter.py
    """
    
    def __init__(self, max_size=100, diversity_threshold=0.7):
        """
        Initializes the genetic reservoir.
        
        Args:
            max_size: Maximum size of the reservoir
            diversity_threshold: Minimum diversity threshold to include new individuals
        """
        self.reservoir = []
        self.max_size = max_size
        self.diversity_threshold = diversity_threshold
        self.crash_types = set()  # Registry of observed crash types
        self.features_cache = {}  # Cache of extracted features
        
        # === SISTEMA PROFESIONAL (INICIALIZACIÓN SILENCIOSA) ===
        self._setup_professional_diversity()
        
    def _setup_professional_diversity(self):
        """Configura el sistema profesional de manera silenciosa"""
        # Determinar librerías disponibles
        self.available_libs = []
        if TEXTDISTANCE_AVAILABLE:
            self.available_libs.append("textdistance")
        if BIOPYTHON_AVAILABLE:
            self.available_libs.append("biopython")
        if SCIPY_AVAILABLE:
            self.available_libs.append("scipy")
        
        # Configurar uso profesional
        self.use_professional_diversity = len(self.available_libs) > 0
        
        # Configurar algoritmos si están disponibles
        self._setup_textdistance_algorithms()
        self._setup_biopython_aligner()
        
        # Estadísticas internas
        self._diversity_stats = {
            "calculations_count": 0,
            "algorithm_used": "original",
            "performance_metrics": {}
        }
        
        # Auto-optimización silenciosa
        self._last_optimization_gen = 0
        self._performance_history = []
    
    def _setup_textdistance_algorithms(self):
        """Configura textdistance silenciosamente"""
        if not TEXTDISTANCE_AVAILABLE:
            self.textdistance_algorithms = None
            return
        
        try:
            self.textdistance_algorithms = {
                'levenshtein': textdistance.Levenshtein(),
                'jaro_winkler': textdistance.JaroWinkler(),
                'jaccard': textdistance.Jaccard(qval=2),
                'cosine': textdistance.Cosine(qval=4),
                'hamming': textdistance.Hamming(),
            }
        except Exception:
            self.textdistance_algorithms = None
    
    def _setup_biopython_aligner(self):
        """Configura BioPython silenciosamente"""
        if not BIOPYTHON_AVAILABLE:
            self.biopython_aligner = None
            return
        
        try:
            self.biopython_aligner = Align.PairwiseAligner()
            self.biopython_aligner.mode = 'local'
            self.biopython_aligner.match_score = 2
            self.biopython_aligner.mismatch_score = -1
            self.biopython_aligner.open_gap_score = -2
            self.biopython_aligner.extend_gap_score = -0.5
        except Exception:
            self.biopython_aligner = None

    def __len__(self):
        """Returns the number of shellcodes in the reservoir."""
        return len(self.reservoir)
        
    def calculate_diversity(self, shellcode1, shellcode2):
        """
        Calculates the diversity between two shellcodes (0-1).
        INTERFAZ ORIGINAL - Usa sistema profesional transparentemente
        
        Args:
            shellcode1: First shellcode (bytes)
            shellcode2: Second shellcode (bytes)
            
        Returns:
            float: Value between 0 (identical) and 1 (completely different)
        """
        self._diversity_stats["calculations_count"] += 1
        
        # Auto-optimización silenciosa cada 100 cálculos
        if self._diversity_stats["calculations_count"] % 100 == 0:
            self._silent_auto_optimization()
        
        # Usar sistema profesional si está disponible
        if self.use_professional_diversity:
            try:
                result = self._calculate_diversity_professional(shellcode1, shellcode2)
                self._diversity_stats["algorithm_used"] = self.available_libs[0] if self.available_libs else "original"
                return result
            except Exception:
                # Fallback silencioso al original
                self.use_professional_diversity = False
        
        # Usar método original como fallback
        result = self._calculate_diversity_original(shellcode1, shellcode2)
        self._diversity_stats["algorithm_used"] = "original"
        return result
    
    def _silent_auto_optimization(self):
        """Auto-optimización silenciosa del threshold de diversidad"""
        if len(self.reservoir) < 10:
            return
        
        # Calcular diversidad promedio actual
        diversities = []
        for i in range(min(10, len(self.reservoir))):
            for j in range(i+1, min(10, len(self.reservoir))):
                div = self._calculate_diversity_professional(self.reservoir[i], self.reservoir[j])
                diversities.append(div)
        
        if not diversities:
            return
        
        avg_diversity = sum(diversities) / len(diversities)
        
        # Ajustar threshold silenciosamente
        if avg_diversity < 0.2:  # Diversidad muy baja
            self.diversity_threshold = max(0.3, self.diversity_threshold - 0.05)
        elif avg_diversity > 0.9 and len(self.reservoir) < self.max_size // 2:  # Diversidad muy alta, reservorio vacío
            self.diversity_threshold = min(0.8, self.diversity_threshold + 0.02)
    
    def _calculate_diversity_professional(self, shellcode1, shellcode2):
        """Sistema de diversidad profesional"""
        if len(shellcode1) == 0 or len(shellcode2) == 0:
            return 1.0
        
        # Estrategia basada en librerías disponibles
        if TEXTDISTANCE_AVAILABLE and self.textdistance_algorithms:
            return self._textdistance_diversity(shellcode1, shellcode2)
        elif BIOPYTHON_AVAILABLE and self.biopython_aligner:
            return self._biopython_diversity(shellcode1, shellcode2)
        elif SCIPY_AVAILABLE:
            return self._scipy_diversity(shellcode1, shellcode2)
        else:
            return self._calculate_diversity_original(shellcode1, shellcode2)
    
    def _textdistance_diversity(self, shellcode1, shellcode2):
        """Diversidad usando textdistance"""
        str1 = shellcode1.hex()
        str2 = shellcode2.hex()
        
        distances = {}
        for name, algo in self.textdistance_algorithms.items():
            try:
                distances[name] = algo.normalized_distance(str1, str2)
            except Exception:
                distances[name] = 1.0
        
        # Pesos adaptativos
        max_len = max(len(shellcode1), len(shellcode2))
        
        if max_len > 1000:
            weights = {'levenshtein': 0.15, 'jaro_winkler': 0.25, 'jaccard': 0.3, 'cosine': 0.25, 'hamming': 0.05}
        elif max_len > 100:
            weights = {'levenshtein': 0.25, 'jaro_winkler': 0.2, 'jaccard': 0.25, 'cosine': 0.2, 'hamming': 0.1}
        else:
            weights = {'levenshtein': 0.35, 'jaro_winkler': 0.25, 'jaccard': 0.2, 'cosine': 0.1, 'hamming': 0.1}
        
        weighted_distance = sum(distances.get(name, 1.0) * weights.get(name, 0.0) for name in weights)
        freq_diversity = self._calculate_frequency_diversity_simple(shellcode1, shellcode2)
        
        return min(1.0, max(0.0, 0.7 * weighted_distance + 0.3 * freq_diversity))
    
    def _biopython_diversity(self, shellcode1, shellcode2):
        """Diversidad usando BioPython"""
        seq1 = ''.join(f'{b:02x}' for b in shellcode1)
        seq2 = ''.join(f'{b:02x}' for b in shellcode2)
        
        try:
            score = self.biopython_aligner.score(seq1, seq2)
            max_possible = min(len(seq1), len(seq2)) * self.biopython_aligner.match_score
            
            if max_possible > 0:
                similarity = score / max_possible
                diversity = 1.0 - max(0.0, min(1.0, similarity))
            else:
                diversity = 1.0
        except Exception:
            diversity = 1.0
        
        freq_diversity = self._calculate_frequency_diversity_simple(shellcode1, shellcode2)
        return 0.6 * diversity + 0.4 * freq_diversity
    
    def _scipy_diversity(self, shellcode1, shellcode2):
        """Diversidad usando SciPy"""
        max_len = max(len(shellcode1), len(shellcode2))
        
        padded1 = list(shellcode1) + [0] * (max_len - len(shellcode1))
        padded2 = list(shellcode2) + [0] * (max_len - len(shellcode2))
        
        try:
            hamming_dist = hamming(padded1, padded2)
            
            set1 = set(shellcode1)
            set2 = set(shellcode2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard_sim = intersection / union if union > 0 else 0
            jaccard_dist = 1 - jaccard_sim
            
            diversity = 0.6 * hamming_dist + 0.4 * jaccard_dist
        except Exception:
            diversity = 1.0
        
        return min(1.0, max(0.0, diversity))
    
    def _calculate_frequency_diversity_simple(self, shellcode1, shellcode2):
        """Análisis de frecuencias simplificado"""
        feat1 = self._extract_simple_features(shellcode1)
        feat2 = self._extract_simple_features(shellcode2)
        
        length_diff = abs(feat1["length"] - feat2["length"]) / max(feat1["length"], feat2["length"])
        freq_diff = self._byte_frequency_difference(shellcode1, shellcode2)
        entropy_diff = abs(feat1["entropy"] - feat2["entropy"]) / max(feat1["entropy"], feat2["entropy"], 0.001)
        
        return 0.4 * freq_diff + 0.3 * length_diff + 0.3 * entropy_diff
    
    def _extract_simple_features(self, shellcode):
        """Extrae características básicas"""
        if not shellcode:
            return {"length": 0, "unique_bytes": 0, "entropy": 0.0}
        
        return {
            "length": len(shellcode),
            "unique_bytes": len(set(shellcode)),
            "entropy": self._calculate_entropy(shellcode)
        }
    
    def _byte_frequency_difference(self, shellcode1, shellcode2):
        """Diferencia en frecuencias de bytes"""
        freq1 = Counter(shellcode1)
        freq2 = Counter(shellcode2)
        
        total1 = len(shellcode1) if shellcode1 else 1
        total2 = len(shellcode2) if shellcode2 else 1
        
        norm_freq1 = {b: count/total1 for b, count in freq1.items()}
        norm_freq2 = {b: count/total2 for b, count in freq2.items()}
        
        all_bytes = set(norm_freq1.keys()) | set(norm_freq2.keys())
        diff = sum(abs(norm_freq1.get(b, 0) - norm_freq2.get(b, 0)) for b in all_bytes)
        
        return min(1.0, diff / 2.0)
    
    def _calculate_entropy(self, data):
        """Entropía de Shannon"""
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
    
    # === MÉTODO ORIGINAL PRESERVADO ===
    def _calculate_diversity_original(self, shellcode1, shellcode2):
        """Método original de diversidad - preservado exactamente"""
        if len(shellcode1) == 0 or len(shellcode2) == 0:
            return 1.0
            
        # Use identifiers for shellcodes
        id1 = self._get_shellcode_id(shellcode1)
        id2 = self._get_shellcode_id(shellcode2)
        
        # If we've already calculated features, use them
        if id1 in self.features_cache and id2 in self.features_cache:
            feat1 = self.features_cache[id1]
            feat2 = self.features_cache[id2]
            
            # Calculate diversity based on features
            instruction_div = self._calculate_instruction_diversity(
                feat1["instruction_types"], 
                feat2["instruction_types"]
            )
            
            # Normalized length difference
            length_diff = abs(feat1["length"] - feat2["length"]) / max(feat1["length"], feat2["length"])
            
            # Syscall difference
            syscall_diff = abs(feat1["syscalls"] - feat2["syscalls"]) / (max(feat1["syscalls"], feat2["syscalls"]) + 1)
            
            # Weights for factors
            return 0.4 * instruction_div + 0.3 * length_diff + 0.3 * syscall_diff
        
        # Fallback method - normalized edit distance
        edit_distance = sum(a != b for a, b in zip(shellcode1[:min(len(shellcode1), len(shellcode2))], 
                                                 shellcode2[:min(len(shellcode1), len(shellcode2))]))
        length_diff = abs(len(shellcode1) - len(shellcode2))
        
        # Normalize to the longer shellcode
        max_len = max(len(shellcode1), len(shellcode2))
        normalized_distance = (edit_distance + length_diff) / max_len if max_len > 0 else 1.0
        
        return normalized_distance
    
    def _calculate_instruction_diversity(self, types1, types2):
        """MÉTODO ORIGINAL preservado exactamente"""
        all_keys = set(types1.keys()) | set(types2.keys())
        
        weights = {
            "known_vulns": 2.5,
            "privileged": 2.0,
            "control_registers": 2.0,
            "speculative_exec": 2.0,
            "syscall": 1.5,
            "memory_access": 1.5,
            "segment_registers": 1.5,
            "forced_exception": 1.5,
            "control_flow": 1.2,
            "stack_manipulation": 1.2,
            "arithmetic": 1.0,
            "simd": 1.0,
            "x86_opcode": 1.0,
            "other": 0.5
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
        """MÉTODO ORIGINAL preservado exactamente"""
        prefix = shellcode[:min(10, len(shellcode))].hex()
        suffix = shellcode[-min(10, len(shellcode)):].hex()
        length = len(shellcode)
        return f"{prefix}_{length}_{suffix}"
    
    def extract_features(self, shellcode):
        """MÉTODO ORIGINAL preservado exactamente"""
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
        """MÉTODO ORIGINAL preservado exactamente"""
        if not self.reservoir:
            return True
            
        self.extract_features(shellcode)
        
        diversities = [self.calculate_diversity(shellcode, existing) 
                      for existing in self.reservoir]
        
        return min(diversities) > self.diversity_threshold
    
    def add(self, shellcode, crash_info=None):
        """MÉTODO ORIGINAL con guardado automático transparente"""
        # Check if the shellcode already exists in the reservoir
        for existing in self.reservoir:
            if shellcode == existing:
                print(f"Shellcode rejected: Duplicate")
                return False
        
        # Always accept if the reservoir is not full
        if len(self.reservoir) < self.max_size:
            if self.is_diverse_enough(shellcode):
                self.reservoir.append(shellcode)
                print(f"Added shellcode to reservoir (size now: {len(self.reservoir)})")
                if crash_info:
                    self.crash_types.add(crash_info.get("crash_type", "unknown"))
                
                # Guardado automático transparente cada 10 adiciones
                if len(self.reservoir) % 10 == 0:
                    self._silent_save()
                
                return True
            else:
                print(f"Shellcode rejected: Not diverse enough")
        else:
            # If the reservoir is full, replace the least diverse
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
                
                # Guardado automático transparente
                if len(self.reservoir) % 10 == 0:
                    self._silent_save()
                
                return True
        
        return False
    
    def _silent_save(self):
        """Guardado silencioso del estado"""
        try:
            self.save_to_file("kernelhunter_reservoir.pkl")
        except Exception:
            pass  # Fallar silenciosamente
    
    def get_sample(self, n=1):
        """MÉTODO ORIGINAL preservado exactamente"""
        if not self.reservoir:
            return []
        return random.sample(self.reservoir, min(n, len(self.reservoir)))
    
    def get_diverse_sample(self, n=1):
        """MÉTODO ORIGINAL preservado exactamente"""
        if len(self.reservoir) <= n:
            return self.reservoir.copy()
            
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
        """MÉTODO ORIGINAL preservado exactamente"""
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
        """MÉTODO ORIGINAL con estadísticas adicionales"""
        import pickle
        
        data = {
            "reservoir": self.reservoir,
            "crash_types": self.crash_types,
            "max_size": self.max_size,
            "diversity_threshold": self.diversity_threshold,
            "_diversity_stats": self._diversity_stats,
            "use_professional_diversity": self.use_professional_diversity,
            "available_libs": self.available_libs
        }
        
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    
    def load_from_file(self, filename):
        """MÉTODO ORIGINAL con soporte para nuevas características"""
        import pickle
        
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                
            self.reservoir = data["reservoir"]
            self.crash_types = data["crash_types"]
            self.max_size = data["max_size"]
            self.diversity_threshold = data["diversity_threshold"]
            
            # Cargar estadísticas si existen
            self._diversity_stats = data.get("_diversity_stats", {
                "calculations_count": 0,
                "algorithm_used": "original",
                "performance_metrics": {}
            })
            
            # Reconfigurar sistema profesional
            self._setup_professional_diversity()
            
            # Recalcular cache
            self.features_cache = {}
            for shellcode in self.reservoir:
                self.extract_features(shellcode)
                
            return True
            
        except (FileNotFoundError, KeyError, pickle.PickleError):
            return False
    
    def _count_syscalls(self, shellcode):
        """MÉTODO ORIGINAL preservado exactamente"""
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
        """MÉTODO ORIGINAL preservado exactamente"""
        privileged_patterns = [
            b"\x0f\x01",  # Various system instructions
            b"\xf4",      # HLT
            b"\x0f\x00",  # Segment control
            b"\x0f\x06",  # CLTS
            b"\x0f\x09",  # WBINVD
            b"\x0f\x30",  # WRMSR
            b"\x0f\x32",  # RDMSR
            b"\x0f\x34",  # SYSENTER
            b"\x0f\x35",  # SYSEXIT
            b"\x0f\x07",  # SYSRET
            b"\x0f\x22",  # MOV CR*, reg
            b"\x0f\x20",  # MOV reg, CR*
            b"\x0f\x01\xd0",  # XGETBV
            b"\x0f\x01\xf8",  # SWAPGS
        ]
        
        count = 0
        for pattern in privileged_patterns:
            count += shellcode.count(pattern)
            
        return count
    
    def _analyze_instruction_types(self, shellcode):
        """MÉTODO ORIGINAL preservado exactamente"""
        types = {
            "syscall": 0,
            "memory_access": 0,
            "privileged": 0,
            "control_flow": 0,
            "arithmetic": 0,
            "simd": 0,
            "segment_registers": 0,
            "speculative_exec": 0,
            "forced_exception": 0,
            "control_registers": 0,
            "stack_manipulation": 0,
            "known_vulns": 0,
            "x86_opcode": 0,
            "other": 0
        }
        
        patterns = {
            "syscall": [b"\x0f\x05", b"\xcd\x80"],
            "memory_access": [
                b"\x48\x8b", b"\x48\x89", b"\xff", b"\x0f\xae", 
                b"\x48\x8a", b"\x48\x88", b"\x48\xa4", b"\x48\xa5",
                b"\x48\xaa", b"\x48\xab"
            ],
            "privileged": [
                b"\x0f\x01", b"\xcd\x80", b"\xf4", b"\x0f\x00", 
                b"\x0f\x06", b"\x0f\x09", b"\x0f\x30", b"\x0f\x32", 
                b"\x0f\x34", b"\x0f\x35", b"\x0f\x07", b"\x0f\x05", 
                b"\x0f\x0b"
            ],
            "control_flow": [
                b"\xe9", b"\xeb", b"\x74", b"\x75", b"\x0f\x84", 
                b"\x0f\x85", b"\xe8", b"\xff", b"\xc3", b"\xc2"
            ],
            "arithmetic": [
                b"\x48\x01", b"\x48\x29", b"\x48\xf7", b"\x48\x0f\xaf", 
                b"\x48\x0f\xc7", b"\x48\x99", b"\x48\xd1"
            ],
            "simd": [
                b"\x0f\x10", b"\x0f\x11", b"\x0f\x28", b"\x0f\x29", 
                b"\x0f\x58", b"\x0f\x59", b"\x0f\x6f", b"\x0f\x7f", 
                b"\x0f\xae", b"\x0f\xc2"
            ],
            "segment_registers": [
                b"\x8e\xd8", b"\x8e\xc0", b"\x8e\xe0", b"\x8e\xe8",
                b"\x8c\xd8", b"\x8c\xc0", b"\x8c\xe0", b"\x8c\xe8"
            ],
            "speculative_exec": [
                b"\x0f\xae\x38", b"\x0f\xae\xf8", b"\x0f\xae\xe8", 
                b"\x0f\xae\xf0", b"\x0f\x31", b"\x0f\xc7\xf8"
            ],
            "forced_exception": [
                b"\x0f\x0b", b"\xcc", b"\xcd\x03", b"\xcd\x04", 
                b"\xf4", b"\xce"
            ],
            "control_registers": [
                b"\x0f\x22\xd8", b"\x0f\x20\xd8", b"\x0f\x22\xe0", 
                b"\x0f\x20\xe0", b"\x0f\x22\xd0", b"\x0f\x20\xd0"
            ],
            "stack_manipulation": [
                b"\x48\x89\xe4", b"\x48\x83\xc4", b"\x48\x83\xec", 
                b"\x48\x8d\x64\x24", b"\x9c", b"\x9d"
            ],
            "known_vulns": [
                b"\x0f\xae\x05", b"\x65\x48\x8b\x04\x25", b"\x0f\x3f", 
                b"\xf3\x0f\xae\xf0", b"\x48\xcf", b"\x0f\x22\xc0", 
                b"\x0f\x32", b"\x0f\x30", b"\x0f\x01\xd0", b"\x0f\x01\xf8", 
                b"\x0f\xae\x38", b"\x0f\x18"
            ]
        }
        
        # Detectar secuencias inteligentemente - evitar doble conteo
        i = 0
        while i < len(shellcode):
            matched = False
            
            # Intentar patrones más largos primero
            for category, pattern_list in sorted(patterns.items(), 
                                                key=lambda x: max([len(p) for p in x[1]], default=0),
                                                reverse=True):
                for pattern in pattern_list:
                    if i <= len(shellcode) - len(pattern) and shellcode[i:i+len(pattern)] == pattern:
                        types[category] += 1
                        i += len(pattern)
                        matched = True
                        break
                if matched:
                    break
            
            # Si no hay patrón, verificar prefijo REX
            if not matched:
                if i < len(shellcode) and 0x40 <= shellcode[i] <= 0x4F:
                    types["x86_opcode"] += 1
                    i += 1
                    instr_len = min(3, len(shellcode) - i)
                    i += instr_len
                else:
                    types["other"] += 1
                    i += 1
        
        return types
    
    def get_diversity_stats(self):
        """MÉTODO ORIGINAL mejorado con estadísticas profesionales"""
        if len(self.reservoir) < 2:
            return {
                "diversity_avg": 0, 
                "diversity_min": 0, 
                "diversity_max": 0,
                "algorithm_used": self._diversity_stats["algorithm_used"],
                "professional_enabled": self.use_professional_diversity,
                "available_libraries": self.available_libs,
                "total_calculations": self._diversity_stats["calculations_count"]
            }
            
        diversities = []
        
        for i in range(len(self.reservoir)):
            for j in range(i+1, len(self.reservoir)):
                diversities.append(self.calculate_diversity(self.reservoir[i], self.reservoir[j]))
        
        instruction_types_counts = Counter()
        for shellcode in self.reservoir:
            shellcode_id = self._get_shellcode_id(shellcode)
            if shellcode_id not in self.features_cache:
                self.extract_features(shellcode)
            
            types = self.features_cache[shellcode_id]["instruction_types"]
            for type_name, count in types.items():
                if count > 0:
                    instruction_types_counts[type_name] += 1
                
        return {
            "diversity_avg": sum(diversities) / len(diversities),
            "diversity_min": min(diversities),
            "diversity_max": max(diversities),
            "unique_crash_types": len(self.crash_types),
            "reservoir_size": len(self.reservoir),
            "avg_shellcode_length": sum(len(sc) for sc in self.reservoir) / len(self.reservoir),
            "instruction_types_distribution": dict(instruction_types_counts.most_common()),
            "algorithm_used": self._diversity_stats["algorithm_used"],
            "professional_enabled": self.use_professional_diversity,
            "available_libraries": self.available_libs,
            "total_calculations": self._diversity_stats["calculations_count"],
            "current_threshold": self.diversity_threshold
        }
    
    def clear_cache(self):
        """MÉTODO ORIGINAL preservado exactamente"""
        self.features_cache = {}
    
    # === MÉTODOS ADICIONALES TRANSPARENTES ===
    def get_professional_status(self):
        """
        Obtiene el estado del sistema profesional.
        MÉTODO ADICIONAL - No interfiere con funcionalidad original.
        """
        return {
            "professional_enabled": self.use_professional_diversity,
            "available_libraries": self.available_libs,
            "primary_algorithm": self.available_libs[0] if self.available_libs else "original",
            "total_calculations": self._diversity_stats["calculations_count"],
            "auto_optimization_active": True
        }
    
    def benchmark_algorithms(self, shellcode1=None, shellcode2=None):
        """
        Compara rendimiento de algoritmos disponibles.
        MÉTODO ADICIONAL - Solo para diagnóstico.
        """
        import time
        
        # Usar shellcodes de prueba si no se proporcionan
        if shellcode1 is None:
            shellcode1 = b"\x48\xc7\xc0\x3c\x00\x00\x00\x48\x31\xff\x0f\x05"
        if shellcode2 is None:
            shellcode2 = b"\x48\xc7\xc0\x01\x00\x00\x00\x48\x31\xff\x0f\x05"
        
        results = {}
        
        # Benchmark original
        start_time = time.time()
        original_result = self._calculate_diversity_original(shellcode1, shellcode2)
        original_time = time.time() - start_time
        results["original"] = {"result": original_result, "time": original_time}
        
        # Benchmark profesional si está disponible
        if self.use_professional_diversity:
            start_time = time.time()
            professional_result = self._calculate_diversity_professional(shellcode1, shellcode2)
            professional_time = time.time() - start_time
            results["professional"] = {"result": professional_result, "time": professional_time}
            results["algorithm_used"] = self.available_libs[0] if self.available_libs else "unknown"
        
        return results
    
    def print_status_report(self):
        """
        Imprime un reporte del estado del sistema.
        MÉTODO ADICIONAL - Solo para diagnóstico.
        """
        print("\n" + "="*50)
        print("🧬 GENETIC RESERVOIR STATUS REPORT")
        print("="*50)
        
        status = self.get_professional_status()
        stats = self.get_diversity_stats() if len(self.reservoir) >= 2 else {}
        
        print(f"📊 Reservoir Size: {len(self.reservoir)}/{self.max_size}")
        print(f"🔬 Professional Mode: {'✅ Enabled' if status['professional_enabled'] else '❌ Disabled'}")
        print(f"🧮 Primary Algorithm: {status['primary_algorithm']}")
        print(f"📚 Available Libraries: {', '.join(status['available_libraries']) if status['available_libraries'] else 'None'}")
        print(f"🔢 Total Calculations: {status['total_calculations']}")
        print(f"⚙️  Current Threshold: {self.diversity_threshold:.3f}")
        
        if stats:
            print(f"📈 Diversity Average: {stats['diversity_avg']:.3f}")
            print(f"📉 Diversity Range: {stats['diversity_min']:.3f} - {stats['diversity_max']:.3f}")
            
            if 'instruction_types_distribution' in stats:
                top_types = list(stats['instruction_types_distribution'].items())[:3]
                print(f"🎯 Top Instruction Types: {top_types}")
        
        print("="*50 + "\n")


# === INICIO AUTOMÁTICO TRANSPARENTE ===
# El sistema se autoconfigura al importar el módulo
def _initialize_system():
    """Inicialización automática del sistema al importar el módulo"""
    if TEXTDISTANCE_AVAILABLE or BIOPYTHON_AVAILABLE or SCIPY_AVAILABLE:
        libs = []
        if TEXTDISTANCE_AVAILABLE:
            libs.append("textdistance")
        if BIOPYTHON_AVAILABLE:
            libs.append("biopython")
        if SCIPY_AVAILABLE:
            libs.append("scipy")
        
        # Solo imprimir si hay librerías profesionales disponibles
        print(f"🧬 GeneticReservoir: Professional diversity system ready ({', '.join(libs)})")
    else:
        # Mensaje opcional - puedes comentar esta línea si quieres que sea completamente silencioso
        print("🧬 GeneticReservoir: Using original diversity algorithm (install textdistance/biopython/scipy for enhanced diversity)")

# Llamar inicialización automática
_initialize_system()


# === INSTRUCCIONES DE USO ===
"""
INSTRUCCIONES DE INSTALACIÓN:

1. REEMPLAZAR ARCHIVO:
   - Hacer backup: cp genetic_reservoir.py genetic_reservoir_original.py
   - Reemplazar completamente genetic_reservoir.py con este código

2. INSTALAR LIBRERÍAS (OPCIONAL):
   - Completa: pip install textdistance[extras] biopython scipy numpy
   - Mínima: pip install textdistance biopython
   - Básica: pip install scipy numpy
   - Sin librerías: El sistema usa el algoritmo original

3. EJECUTAR KERNELHUNTER:
   - python kernelhunter.py
   - NO se requieren modificaciones en kernelhunter.py
   - El sistema es 100% transparente y compatible

4. VERIFICAR FUNCIONAMIENTO:
   - Al iniciar verás: "🧬 GeneticReservoir: Professional diversity system ready"
   - Si no tienes librerías: "🧬 GeneticReservoir: Using original diversity algorithm"

5. DIAGNÓSTICO (OPCIONAL):
   - En Python: reservoir.print_status_report()
   - En Python: reservoir.benchmark_algorithms()
   - En Python: reservoir.get_professional_status()

CARACTERÍSTICAS AUTOMÁTICAS:
✅ Auto-detección de librerías disponibles
✅ Fallback graceful al algoritmo original
✅ Auto-optimización de thresholds de diversidad
✅ Guardado automático cada 10 adiciones al reservorio
✅ Compatibilidad 100% con kernelhunter.py existente
✅ Mejora de diversidad transparente
✅ Preservación de todos los métodos originales

BENEFICIOS:
🎯 Diversidad real entre shellcodes largos
🚀 Múltiples linajes evolutivos paralelos
🔧 Auto-ajuste de parámetros según rendimiento
💾 Persistencia automática del estado
📊 Estadísticas avanzadas disponibles
🛡️ Funciona con o sin librerías adicionales
"""
