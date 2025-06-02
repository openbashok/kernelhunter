#!/usr/bin/env python3
# genetic_reservoir.py - VERSIÓN DE DIAGNÓSTICO
# Esta versión tiene prints para encontrar dónde se cuelga

import random
import numpy as np
from collections import Counter

print("DEBUG: Starting genetic_reservoir import...")

# === IMPORTS SEGUROS CON FALLBACKS ===
try:
    import textdistance
    TEXTDISTANCE_AVAILABLE = True
    print("DEBUG: textdistance imported successfully")
except ImportError:
    TEXTDISTANCE_AVAILABLE = False
    print("DEBUG: textdistance not available")

try:
    from Bio import Align
    BIOPYTHON_AVAILABLE = True
    print("DEBUG: biopython imported successfully")
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("DEBUG: biopython not available")

try:
    from scipy.spatial.distance import hamming
    SCIPY_AVAILABLE = True
    print("DEBUG: scipy imported successfully")
except ImportError:
    SCIPY_AVAILABLE = False
    print("DEBUG: scipy not available")

print("DEBUG: All imports completed")

class GeneticReservoir:
    """
    Implements a genetic reservoir to maintain diversity in the population.
    VERSIÓN DE DIAGNÓSTICO para encontrar problemas.
    """
    
    def __init__(self, max_size=100, diversity_threshold=0.7):
        """
        Initializes the genetic reservoir.
        """
        print("DEBUG: GeneticReservoir.__init__ starting...")
        
        self.reservoir = []
        self.max_size = max_size
        self.diversity_threshold = diversity_threshold
        self.crash_types = set()
        self.features_cache = {}
        
        print("DEBUG: Basic attributes set, calling _safe_setup...")
        
        # === SISTEMA PROFESIONAL DESHABILITADO TEMPORALMENTE ===
        self.use_enhanced_diversity = False
        self.textdistance_algo = None
        self._calculation_count = 0
        
        print("DEBUG: GeneticReservoir.__init__ completed successfully")
        
    def __len__(self):
        """Returns the number of shellcodes in the reservoir."""
        return len(self.reservoir)
        
    def calculate_diversity(self, shellcode1, shellcode2):
        """
        Calculates the diversity between two shellcodes (0-1).
        VERSIÓN SIMPLIFICADA PARA DIAGNÓSTICO
        """
        # Solo usar método original por ahora
        return self._calculate_diversity_original(shellcode1, shellcode2)
    
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
        """Generates a unique identifier for a shellcode."""
        prefix = shellcode[:min(10, len(shellcode))].hex()
        suffix = shellcode[-min(10, len(shellcode)):].hex()
        length = len(shellcode)
        return f"{prefix}_{length}_{suffix}"
    
    def extract_features(self, shellcode):
        """Extract features - VERSIÓN SIMPLIFICADA PARA DIAGNÓSTICO"""
        print(f"DEBUG: extract_features called for shellcode of length {len(shellcode)}")
        
        shellcode_id = self._get_shellcode_id(shellcode)
        if shellcode_id in self.features_cache:
            print("DEBUG: features found in cache")
            return self.features_cache[shellcode_id]
        
        print("DEBUG: extracting new features...")
        
        # Extract features
        features = {
            "length": len(shellcode),
            "syscalls": self._count_syscalls(shellcode),
            "privileged_instr": self._count_privileged_instructions(shellcode),
            "instruction_types": self._analyze_instruction_types(shellcode),
        }
        
        print("DEBUG: features extracted successfully")
        
        # Save to cache
        self.features_cache[shellcode_id] = features
        return features
    
    def is_diverse_enough(self, shellcode):
        """Check diversity - CON DIAGNÓSTICO"""
        print(f"DEBUG: is_diverse_enough called, reservoir size: {len(self.reservoir)}")
        
        if not self.reservoir:
            print("DEBUG: reservoir empty, returning True")
            return True
        
        print("DEBUG: extracting features for new shellcode...")
        self.extract_features(shellcode)
        
        print("DEBUG: calculating diversity against existing shellcodes...")
        diversities = []
        for i, existing in enumerate(self.reservoir):
            print(f"DEBUG: calculating diversity against shellcode {i+1}/{len(self.reservoir)}")
            div = self.calculate_diversity(shellcode, existing)
            diversities.append(div)
            print(f"DEBUG: diversity {i+1}: {div:.3f}")
        
        min_diversity = min(diversities)
        result = min_diversity > self.diversity_threshold
        
        print(f"DEBUG: min_diversity: {min_diversity:.3f}, threshold: {self.diversity_threshold:.3f}, diverse_enough: {result}")
        
        return result
    
    def add(self, shellcode, crash_info=None):
        """Add shellcode - CON DIAGNÓSTICO DETALLADO"""
        print(f"DEBUG: add() called with shellcode of length {len(shellcode)}")
        
        # Check if the shellcode already exists in the reservoir
        print("DEBUG: checking for duplicates...")
        for i, existing in enumerate(self.reservoir):
            if shellcode == existing:
                print(f"DEBUG: duplicate found at index {i}")
                print(f"Shellcode rejected: Duplicate")
                return False
        
        print("DEBUG: no duplicates found")
        
        # Always accept if the reservoir is not full
        if len(self.reservoir) < self.max_size:
            print(f"DEBUG: reservoir not full ({len(self.reservoir)}/{self.max_size}), checking diversity...")
            
            if self.is_diverse_enough(shellcode):
                print("DEBUG: shellcode is diverse enough, adding to reservoir...")
                self.reservoir.append(shellcode)
                print(f"Added shellcode to reservoir (size now: {len(self.reservoir)})")
                if crash_info:
                    self.crash_types.add(crash_info.get("crash_type", "unknown"))
                print("DEBUG: add() completed successfully")
                return True
            else:
                print(f"Shellcode rejected: Not diverse enough")
                print("DEBUG: add() completed - not diverse enough")
                return False
        else:
            print(f"DEBUG: reservoir full ({len(self.reservoir)}/{self.max_size}), checking for replacement...")
            
            if self.is_diverse_enough(shellcode):
                print("DEBUG: calculating diversity scores for replacement...")
                
                # Find the shellcode most similar to others (least diverse)
                diversity_scores = []
                for i, sc in enumerate(self.reservoir):
                    print(f"DEBUG: calculating avg diversity for shellcode {i+1}/{len(self.reservoir)}")
                    
                    diversities_for_this = []
                    for j, other in enumerate(self.reservoir):
                        if i != j:
                            div = self.calculate_diversity(sc, other)
                            diversities_for_this.append(div)
                    
                    avg_diversity = sum(diversities_for_this) / len(diversities_for_this)
                    diversity_scores.append((i, avg_diversity))
                    print(f"DEBUG: avg diversity for shellcode {i}: {avg_diversity:.3f}")
                
                # Replace the least diverse
                least_diverse_idx = min(diversity_scores, key=lambda x: x[1])[0]
                print(f"DEBUG: replacing shellcode at index {least_diverse_idx}")
                
                self.reservoir[least_diverse_idx] = shellcode
                
                if crash_info:
                    self.crash_types.add(crash_info.get("crash_type", "unknown"))
                
                print("DEBUG: replacement completed successfully")
                return True
            else:
                print("DEBUG: not diverse enough for replacement")
                return False
    
    def get_sample(self, n=1):
        """Gets a random sample from the reservoir."""
        if not self.reservoir:
            return []
        return random.sample(self.reservoir, min(n, len(self.reservoir)))
    
    def get_diverse_sample(self, n=1):
        """Gets a diverse sample from the reservoir."""
        print(f"DEBUG: get_diverse_sample called with n={n}, reservoir size={len(self.reservoir)}")
        
        if len(self.reservoir) <= n:
            return self.reservoir.copy()
            
        # Selection based on maximum diversity
        selected = [random.choice(self.reservoir)]
        print(f"DEBUG: selected initial shellcode")
        
        while len(selected) < n:
            print(f"DEBUG: selecting shellcode {len(selected)+1}/{n}")
            
            # Calculate maximum diversity for each candidate
            max_diversities = []
            
            for i, candidate in enumerate(self.reservoir):
                if candidate in selected:
                    continue
                    
                # Minimum diversity with respect to those already selected
                min_divs = []
                for s in selected:
                    div = self.calculate_diversity(candidate, s)
                    min_divs.append(div)
                
                min_div = min(min_divs)
                max_diversities.append((candidate, min_div))
            
            # Select the candidate with the highest minimum diversity
            if max_diversities:
                next_selection = max(max_diversities, key=lambda x: x[1])[0]
                selected.append(next_selection)
                print(f"DEBUG: selected shellcode with min_div: {max(max_diversities, key=lambda x: x[1])[1]:.3f}")
            else:
                print("DEBUG: no more candidates available")
                break
        
        print(f"DEBUG: get_diverse_sample completed, returning {len(selected)} shellcodes")
        return selected
    
    def get_by_feature(self, feature_name, value, comparison="gt", limit=5):
        """Gets shellcodes that match a feature criterion."""
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
        """Saves the genetic reservoir to a file."""
        print(f"DEBUG: save_to_file called with filename: {filename}")
        
        import pickle
        
        data = {
            "reservoir": self.reservoir,
            "crash_types": self.crash_types,
            "max_size": self.max_size,
            "diversity_threshold": self.diversity_threshold
        }
        
        try:
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            print(f"DEBUG: save_to_file completed successfully")
        except Exception as e:
            print(f"DEBUG: save_to_file failed: {e}")
    
    def load_from_file(self, filename):
        """Loads the genetic reservoir from a file."""
        print(f"DEBUG: load_from_file called with filename: {filename}")
        
        import pickle
        
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                
            self.reservoir = data["reservoir"]
            self.crash_types = data["crash_types"]
            self.max_size = data["max_size"]
            self.diversity_threshold = data["diversity_threshold"]
            
            print(f"DEBUG: loaded {len(self.reservoir)} shellcodes from file")
            
            # Recalculate feature cache
            print("DEBUG: recalculating feature cache...")
            self.features_cache = {}
            for i, shellcode in enumerate(self.reservoir):
                print(f"DEBUG: extracting features for loaded shellcode {i+1}/{len(self.reservoir)}")
                self.extract_features(shellcode)
            
            print("DEBUG: load_from_file completed successfully")
            return True
            
        except (FileNotFoundError, KeyError, pickle.PickleError) as e:
            print(f"DEBUG: load_from_file failed: {e}")
            return False
    
    def _count_syscalls(self, shellcode):
        """Counts the system calls in the shellcode."""
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
        """Counts privileged instructions in the shellcode."""
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
        """Analyzes the types of instructions - VERSIÓN SIMPLIFICADA"""
        print(f"DEBUG: _analyze_instruction_types called for shellcode of length {len(shellcode)}")
        
        # Versión simplificada para evitar loops complejos
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
        
        # Contar solo patrones básicos por ahora
        if b"\x0f\x05" in shellcode or b"\xcd\x80" in shellcode:
            types["syscall"] += 1
        
        if b"\x48\x8b" in shellcode or b"\x48\x89" in shellcode:
            types["memory_access"] += 1
            
        if b"\x0f\x01" in shellcode or b"\xf4" in shellcode:
            types["privileged"] += 1
        
        # Contar otros como "other"
        types["other"] = max(1, len(shellcode) // 10)
        
        print(f"DEBUG: _analyze_instruction_types completed")
        return types
    
    def get_diversity_stats(self):
        """Gets diversity statistics for the reservoir."""
        if len(self.reservoir) < 2:
            return {"diversity_avg": 0, "diversity_min": 0, "diversity_max": 0}
            
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
            "instruction_types_distribution": dict(instruction_types_counts.most_common())
        }
    
    def clear_cache(self):
        """Clears the feature cache to save memory."""
        self.features_cache = {}


# === INICIALIZACIÓN CON DIAGNÓSTICO ===
print("DEBUG: genetic_reservoir module initialization...")

if TEXTDISTANCE_AVAILABLE:
    print("🧬 GeneticReservoir: Professional diversity system ready (textdistance)")
elif BIOPYTHON_AVAILABLE:
    print("🧬 GeneticReservoir: Professional diversity system ready (biopython)")
elif SCIPY_AVAILABLE:
    print("🧬 GeneticReservoir: Professional diversity system ready (scipy)")
else:
    print("🧬 GeneticReservoir: Using original diversity algorithm")

print("DEBUG: genetic_reservoir module loaded successfully")
