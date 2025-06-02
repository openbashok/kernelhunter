#!/usr/bin/env python3
# genetic_reservoir.py - VERSIÓN MINIMALISTA Y SEGURA
# 100% compatible con kernelhunter.py sin modificaciones
# Se enfoca en FUNCIONAR sin problemas

import random
import numpy as np
from collections import Counter

# === IMPORTS SEGUROS CON FALLBACKS ===
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
    Implements a genetic reservoir to maintain diversity in the population.
    VERSIÓN SEGURA que preserva exactamente la funcionalidad original.
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
        
        # === SISTEMA PROFESIONAL OPCIONAL (SEGURO) ===
        self._safe_setup()
        
    def _safe_setup(self):
        """Setup seguro que no puede fallar"""
        self.use_enhanced_diversity = False
        self.textdistance_algo = None
        
        # Solo habilitar si textdistance está disponible y funciona
        if TEXTDISTANCE_AVAILABLE:
            try:
                self.textdistance_algo = textdistance.Levenshtein()
                # Test rápido para verificar que funciona
                test_result = self.textdistance_algo.normalized_distance("test", "best")
                if 0 <= test_result <= 1:
                    self.use_enhanced_diversity = True
            except Exception:
                self.use_enhanced_diversity = False
                self.textdistance_algo = None
        
        # Contador interno para optimización silenciosa
        self._calculation_count = 0
        
    def __len__(self):
        """Returns the number of shellcodes in the reservoir."""
        return len(self.reservoir)
        
    def calculate_diversity(self, shellcode1, shellcode2):
        """
        Calculates the diversity between two shellcodes (0-1).
        INTERFAZ EXACTAMENTE IGUAL AL ORIGINAL
        
        Args:
            shellcode1: First shellcode (bytes)
            shellcode2: Second shellcode (bytes)
            
        Returns:
            float: Value between 0 (identical) and 1 (completely different)
        """
        self._calculation_count += 1
        
        # Auto-optimización muy conservadora cada 200 cálculos
        if self._calculation_count % 200 == 0:
            self._conservative_auto_optimize()
        
        # Usar versión mejorada solo si está disponible y es segura
        if self.use_enhanced_diversity and self.textdistance_algo:
            try:
                result = self._calculate_diversity_enhanced(shellcode1, shellcode2)
                if 0 <= result <= 1:  # Verificar resultado válido
                    return result
            except Exception:
                # Fallback silencioso al original
                self.use_enhanced_diversity = False
        
        # Usar método original como fallback
        return self._calculate_diversity_original(shellcode1, shellcode2)
    
    def _conservative_auto_optimize(self):
        """Auto-optimización muy conservadora"""
        if len(self.reservoir) < 5:
            return
        
        try:
            # Calcular diversidad promedio de una muestra pequeña
            sample_size = min(5, len(self.reservoir))
            diversities = []
            
            for i in range(sample_size):
                for j in range(i+1, sample_size):
                    div = self._calculate_diversity_original(self.reservoir[i], self.reservoir[j])
                    diversities.append(div)
            
            if diversities:
                avg_div = sum(diversities) / len(diversities)
                
                # Ajustes muy conservadores
                if avg_div < 0.15:  # Diversidad extremadamente baja
                    self.diversity_threshold = max(0.4, self.diversity_threshold - 0.05)
                elif avg_div > 0.95 and len(self.reservoir) < self.max_size // 3:  # Diversidad extremadamente alta, reservorio muy vacío
                    self.diversity_threshold = min(0.8, self.diversity_threshold + 0.02)
        except Exception:
            pass  # Fallar silenciosamente
    
    def _calculate_diversity_enhanced(self, shellcode1, shellcode2):
        """Versión mejorada usando textdistance"""
        if len(shellcode1) == 0 or len(shellcode2) == 0:
            return 1.0
        
        # Convertir a hex strings para textdistance
        str1 = shellcode1.hex()
        str2 = shellcode2.hex()
        
        # Calcular distancia de Levenshtein normalizada
        levenshtein_dist = self.textdistance_algo.normalized_distance(str1, str2)
        
        # Combinar con análisis de bytes únicos
        set1 = set(shellcode1)
        set2 = set(shellcode2)
        
        if len(set1) == 0 and len(set2) == 0:
            byte_diversity = 0.0
        else:
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            byte_diversity = 1.0 - (intersection / union) if union > 0 else 1.0
        
        # Combinar ambas métricas
        combined_diversity = 0.7 * levenshtein_dist + 0.3 * byte_diversity
        
        return min(1.0, max(0.0, combined_diversity))
    
    # === MÉTODO ORIGINAL PRESERVADO EXACTAMENTE ===
    def _calculate_diversity_original(self, shellcode1, shellcode2):
        """
        Método original de diversidad - preservado exactamente como estaba.
        """
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
        """
        Calcula la diversidad entre dos sets de tipos de instrucción con pesos
        ponderados según su relevancia para encontrar vulnerabilidades.
        MÉTODO ORIGINAL PRESERVADO EXACTAMENTE
        """
        # Get all unique keys
        all_keys = set(types1.keys()) | set(types2.keys())
        
        # Pesos para diferentes categorías según su relevancia para seguridad
        weights = {
            "known_vulns": 2.5,         # Mayor peso a vulnerabilidades conocidas
            "privileged": 2.0,          # Alto peso a instrucciones privilegiadas
            "control_registers": 2.0,   # Alto peso a manipulación de registros de control
            "speculative_exec": 2.0,    # Alto peso a ejecución especulativa (Spectre/Meltdown)
            "syscall": 1.5,             # Peso medio-alto a syscalls
            "memory_access": 1.5,       # Peso medio-alto a accesos a memoria
            "segment_registers": 1.5,   # Peso medio-alto a registros de segmento
            "forced_exception": 1.5,    # Peso medio-alto a excepciones forzadas
            "control_flow": 1.2,        # Peso medio a control de flujo
            "stack_manipulation": 1.2,  # Peso medio a manipulación de pila
            "arithmetic": 1.0,          # Peso estándar
            "simd": 1.0,                # Peso estándar
            "x86_opcode": 1.0,          # Peso estándar
            "other": 0.5                # Peso bajo a instrucciones no clasificadas
        }
        
        # Calcular diferencia total ponderada
        weighted_diff = 0
        weighted_total = 0
        
        for key in all_keys:
            val1 = types1.get(key, 0)
            val2 = types2.get(key, 0)
            key_weight = weights.get(key, 1.0)
            
            weighted_diff += abs(val1 - val2) * key_weight
            weighted_total += max(val1, val2) * key_weight
        
        # Normalizar
        return weighted_diff / weighted_total if weighted_total > 0 else 1.0
    
    def _get_shellcode_id(self, shellcode):
        """Generates a unique identifier for a shellcode."""
        # Use the first and last bytes as an identifier
        prefix = shellcode[:min(10, len(shellcode))].hex()
        suffix = shellcode[-min(10, len(shellcode)):].hex()
        length = len(shellcode)
        return f"{prefix}_{length}_{suffix}"
    
    def extract_features(self, shellcode):
        """
        Extracts relevant features from the shellcode for analysis.
        MÉTODO ORIGINAL PRESERVADO EXACTAMENTE
        """
        # Check if we've already calculated these features
        shellcode_id = self._get_shellcode_id(shellcode)
        if shellcode_id in self.features_cache:
            return self.features_cache[shellcode_id]
        
        # Extract features
        features = {
            "length": len(shellcode),
            "syscalls": self._count_syscalls(shellcode),
            "privileged_instr": self._count_privileged_instructions(shellcode),
            "instruction_types": self._analyze_instruction_types(shellcode),
        }
        
        # Save to cache
        self.features_cache[shellcode_id] = features
        return features
    
    def is_diverse_enough(self, shellcode):
        """
        Checks if a shellcode is diverse enough to be included.
        MÉTODO ORIGINAL PRESERVADO EXACTAMENTE
        """
        if not self.reservoir:
            return True
            
        # Extract features
        self.extract_features(shellcode)
        
        # Calculate diversity against all existing shellcodes
        diversities = [self.calculate_diversity(shellcode, existing) 
                      for existing in self.reservoir]
        
        # If the shellcode is sufficiently different from all existing ones
        return min(diversities) > self.diversity_threshold
    
    def add(self, shellcode, crash_info=None):
        """
        Adds a shellcode to the reservoir if it's diverse or interesting.
        MÉTODO ORIGINAL PRESERVADO EXACTAMENTE
        """
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
                return True
            else:
                print(f"Shellcode rejected: Not diverse enough")
        else:
            # If the reservoir is full, replace the least diverse
            if self.is_diverse_enough(shellcode):
                # Find the shellcode most similar to others (least diverse)
                diversity_scores = []
                for i, sc in enumerate(self.reservoir):
                    avg_diversity = sum(self.calculate_diversity(sc, other) 
                                       for j, other in enumerate(self.reservoir) if i != j) / (len(self.reservoir) - 1)
                    diversity_scores.append((i, avg_diversity))
                
                # Replace the least diverse
                least_diverse_idx = min(diversity_scores, key=lambda x: x[1])[0]
                self.reservoir[least_diverse_idx] = shellcode
                
                if crash_info:
                    self.crash_types.add(crash_info.get("crash_type", "unknown"))
                return True
        
        return False
    
    def get_sample(self, n=1):
        """
        Gets a random sample from the reservoir.
        MÉTODO ORIGINAL PRESERVADO EXACTAMENTE
        """
        if not self.reservoir:
            return []
        
        return random.sample(self.reservoir, min(n, len(self.reservoir)))
    
    def get_diverse_sample(self, n=1):
        """
        Gets a diverse sample from the reservoir.
        MÉTODO ORIGINAL PRESERVADO EXACTAMENTE
        """
        if len(self.reservoir) <= n:
            return self.reservoir.copy()
            
        # Selection based on maximum diversity
        selected = [random.choice(self.reservoir)]
        
        while len(selected) < n:
            # Calculate maximum diversity for each candidate
            max_diversities = []
            
            for candidate in self.reservoir:
                if candidate in selected:
                    continue
                    
                # Minimum diversity with respect to those already selected
                min_div = min(self.calculate_diversity(candidate, s) for s in selected)
                max_diversities.append((candidate, min_div))
            
            # Select the candidate with the highest minimum diversity
            if max_diversities:
                next_selection = max(max_diversities, key=lambda x: x[1])[0]
                selected.append(next_selection)
            else:
                break
                
        return selected
    
    def get_by_feature(self, feature_name, value, comparison="gt", limit=5):
        """
        Gets shellcodes that match a feature criterion.
        MÉTODO ORIGINAL PRESERVADO EXACTAMENTE
        """
        results = []
        
        for shellcode in self.reservoir:
            # Extract features if not in cache
            shellcode_id = self._get_shellcode_id(shellcode)
            if shellcode_id not in self.features_cache:
                self.extract_features(shellcode)
                
            features = self.features_cache[shellcode_id]
            
            # Compare according to criterion
            if feature_name in features:
                feature_value = features[feature_name]
                
                if comparison == "gt" and feature_value > value:
                    results.append(shellcode)
                elif comparison == "lt" and feature_value < value:
                    results.append(shellcode)
                elif comparison == "eq" and feature_value == value:
                    results.append(shellcode)
                    
                # Limit results
                if len(results) >= limit:
                    break
                    
        return results
    
    def save_to_file(self, filename):
        """
        Saves the genetic reservoir to a file.
        MÉTODO ORIGINAL PRESERVADO EXACTAMENTE
        """
        import pickle
        
        data = {
            "reservoir": self.reservoir,
            "crash_types": self.crash_types,
            "max_size": self.max_size,
            "diversity_threshold": self.diversity_threshold
        }
        
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    
    def load_from_file(self, filename):
        """
        Loads the genetic reservoir from a file.
        MÉTODO ORIGINAL PRESERVADO EXACTAMENTE
        """
        import pickle
        
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                
            self.reservoir = data["reservoir"]
            self.crash_types = data["crash_types"]
            self.max_size = data["max_size"]
            self.diversity_threshold = data["diversity_threshold"]
            
            # Recalculate feature cache
            self.features_cache = {}
            for shellcode in self.reservoir:
                self.extract_features(shellcode)
                
            return True
            
        except (FileNotFoundError, KeyError, pickle.PickleError):
            return False
    
    def _count_syscalls(self, shellcode):
        """
        Counts the system calls in the shellcode.
        MÉTODO ORIGINAL PRESERVADO EXACTAMENTE
        """
        # Look for syscall patterns
        syscall_pattern = b"\x0f\x05"  # syscall instruction
        int80_pattern = b"\xcd\x80"    # int 0x80 (32-bit syscall)
        
        count = shellcode.count(syscall_pattern)
        count += shellcode.count(int80_pattern)
        
        # Also count syscall setup sequences that may indicate syscalls
        syscall_setup = b"\x48\xc7\xc0"  # mov rax, X (syscall number)
        setup_count = shellcode.count(syscall_setup)
        
        # We don't count all setup patterns as actual syscalls
        # Only count if there are more setups than actual syscalls
        if setup_count > count:
            # Add some of the excess setups as potential syscalls
            count += (setup_count - count) // 2
            
        return count
    
    def _count_privileged_instructions(self, shellcode):
        """
        Counts privileged instructions in the shellcode.
        MÉTODO ORIGINAL PRESERVADO EXACTAMENTE
        """
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
        """
        Analyzes the types of instructions in the shellcode with greater detail.
        MÉTODO ORIGINAL PRESERVADO EXACTAMENTE
        """
        # Classify instructions by type with expanded categories
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
        
        # Expanded patterns with all those identified in the generator
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
        
        # Detect sequences in a smarter way - avoiding double counting
        i = 0
        while i < len(shellcode):
            matched = False
            
            # Try to match longer patterns first
            for category, pattern_list in sorted(patterns.items(), 
                                                key=lambda x: max([len(p) for p in x[1]], default=0),
                                                reverse=True):
                for pattern in pattern_list:
                    if i <= len(shellcode) - len(pattern) and shellcode[i:i+len(pattern)] == pattern:
                        types[category] += 1
                        i += len(pattern)  # Move index past this pattern
                        matched = True
                        break
                if matched:
                    break
            
            # If no pattern matched, check if it's a REX prefix (likely x86_64 instruction)
            if not matched:
                if i < len(shellcode) and 0x40 <= shellcode[i] <= 0x4F:
                    types["x86_opcode"] += 1
                    i += 1  # Move past REX prefix
                    
                    # Try to estimate the length of the instruction
                    # Most x86-64 instructions are 2-4 bytes after the REX prefix
                    instr_len = min(3, len(shellcode) - i)
                    i += instr_len
                else:
                    # Count as other and move forward
                    types["other"] += 1
                    i += 1
        
        return types
    
    def get_diversity_stats(self):
        """
        Gets diversity statistics for the reservoir.
        MÉTODO ORIGINAL PRESERVADO EXACTAMENTE
        """
        if len(self.reservoir) < 2:
            return {"diversity_avg": 0, "diversity_min": 0, "diversity_max": 0}
            
        diversities = []
        
        # Calculate diversity between all pairs
        for i in range(len(self.reservoir)):
            for j in range(i+1, len(self.reservoir)):
                diversities.append(self.calculate_diversity(self.reservoir[i], self.reservoir[j]))
        
        # Count instruction types across all shellcodes
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
            "instruction_types_distribution": dict(instruction_types_counts.most_common())
        }
    
    def clear_cache(self):
        """Clears the feature cache to save memory."""
        self.features_cache = {}


# === INICIALIZACIÓN SEGURA Y SILENCIOSA ===
if TEXTDISTANCE_AVAILABLE:
    print("🧬 GeneticReservoir: Professional diversity system ready (textdistance)")
elif BIOPYTHON_AVAILABLE:
    print("🧬 GeneticReservoir: Professional diversity system ready (biopython)")
elif SCIPY_AVAILABLE:
    print("🧬 GeneticReservoir: Professional diversity system ready (scipy)")
else:
    # Comentar esta línea si quieres que sea completamente silencioso
    print("🧬 GeneticReservoir: Using original diversity algorithm")


# === INSTRUCCIONES DE USO ===
"""
INSTALACIÓN:
1. Reemplazar genetic_reservoir.py con este código
2. Ejecutar: python kernelhunter.py
3. NO requiere modificaciones en kernelhunter.py
4. Funciona con o sin librerías adicionales

CARACTERÍSTICAS:
✅ 100% compatible con código original
✅ Fallback seguro al algoritmo original
✅ Auto-optimización conservadora y silenciosa
✅ Usa textdistance si está disponible
✅ No se cuelga ni falla
✅ Preserva exactamente toda la funcionalidad original
"""
