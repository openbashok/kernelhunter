#!/usr/bin/env python3
import curses
import json
import os
import time
import pickle
import re
from collections import defaultdict, Counter
from datetime import datetime
try:
    from kernelhunter_config import get_reservoir_file
except Exception:
    def get_reservoir_file(name="kernelhunter_reservoir.pkl"):
        return name

# Key files and paths
METRICS_FILE = "kernelhunter_metrics.json"
CRASH_LOG = "kernelhunter_crashes.txt" 
LOG_FILE = "kernelhunter_survivors.txt"
RESERVOIR_FILE = get_reservoir_file()
CRASH_DIR = "kernelhunter_critical"
OUTPUT_DIR = "kernelhunter_generations"

class KernelHunterMonitor:
    def __init__(self):
        # Metrics storage
        self.metrics = {
            "generations": [],
            "crash_rates": [],
            "system_impacts": [],
            "shellcode_lengths": [],
            "crash_types": {},
            "latest_gen": 0,
            "population_size": 0,
            "reservoir_size": 0,
            "reservoir_diversity": 0.0,
            "critical_crashes": 0,
            "instruction_diversity": {},
            "system_health": "Good",
            "last_updated": None,
            "genetic_diversity": 0.0
        }
        
        # Initialize counters
        self.refresh_count = 0
        self.last_refresh = time.time()
        self.start_time = time.time()
        
        # Read reservoir if exists
        self.genetic_reservoir = None
        self.reservoir_stats = {}
    
    def load_reservoir(self):
        """Load genetic reservoir if available"""
        try:
            if os.path.exists(RESERVOIR_FILE):
                from genetic_reservoir import GeneticReservoir

                reservoir = GeneticReservoir()
                if reservoir.load_from_file(RESERVOIR_FILE):
                    self.genetic_reservoir = reservoir
                    self.metrics["reservoir_size"] = len(reservoir)

                    try:
                        stats = reservoir.get_diversity_stats()
                        self.reservoir_stats = stats
                        self.metrics["reservoir_diversity"] = stats.get("diversity_avg", 0.0)
                        self.metrics["instruction_diversity"] = stats.get("instruction_types_distribution", {})
                    except Exception:
                        pass
                    return True
                else:
                    # Fallback to direct pickle loading (older format)
                    with open(RESERVOIR_FILE, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, dict):
                        self.metrics["reservoir_size"] = len(data.get("reservoir", []))
                        self.genetic_reservoir = None
                        return True
        except Exception:
            return False
        return False
    
    def load_metrics(self):
        """Load metrics from the metrics file"""
        try:
            if os.path.exists(METRICS_FILE):
                with open(METRICS_FILE, 'r') as f:
                    data = json.load(f)
                    
                # Update our metrics with the file data
                for key, value in data.items():
                    if key in self.metrics:
                        self.metrics[key] = value
                
                # Set latest generation
                if self.metrics["generations"]:
                    self.metrics["latest_gen"] = max(self.metrics["generations"])
                
                return True
        except Exception as e:
            return False
        return False
    
    def count_critical_crashes(self):
        """Count critical crash files"""
        if os.path.exists(CRASH_DIR):
            crash_files = [f for f in os.listdir(CRASH_DIR) if f.endswith(".json")]
            self.metrics["critical_crashes"] = len(crash_files)
    
    def estimate_population_size(self):
        """Estimate current population size by checking latest generation directory"""
        try:
            if self.metrics["latest_gen"] > 0:
                gen_dir = f"gen_{self.metrics['latest_gen']:04d}"
                full_path = os.path.join(OUTPUT_DIR, gen_dir)
                
                if os.path.exists(full_path):
                    # Count binary files (executables without extension)
                    prog_count = len([f for f in os.listdir(full_path) 
                                     if f.startswith('g') and not f.endswith('.c')])
                    
                    # Count survivor programs from log
                    if os.path.exists(LOG_FILE):
                        with open(LOG_FILE, 'r') as f:
                            log_content = f.read()
                            latest_gen_section = re.search(rf"\[GEN {self.metrics['latest_gen']}\].*?(?=\[GEN|\Z)", 
                                                         log_content, re.DOTALL)
                            
                            if latest_gen_section:
                                survivor_count = len(re.findall(r"Survivor \d+:", latest_gen_section.group(0)))
                                if survivor_count > 0:
                                    self.metrics["population_size"] = survivor_count
                                    return
            
            # Fallback: use the number of programs per generation
            self.metrics["population_size"] = self.estimate_programs_per_generation()
        except Exception:
            self.metrics["population_size"] = 0
    
    def estimate_programs_per_generation(self):
        """Estimate the number of programs per generation based on files"""
        try:
            if self.metrics["latest_gen"] > 0:
                gen_dir = f"gen_{self.metrics['latest_gen']:04d}"
                full_path = os.path.join(OUTPUT_DIR, gen_dir)
                
                if os.path.exists(full_path):
                    files = os.listdir(full_path)
                    return len(files) // 2  # Each program has .c file and binary
            
            return 0
        except Exception:
            return 0
    
    def scan_crash_log(self):
        """Scan the crash log file for latest crash information"""
        try:
            if os.path.exists(CRASH_LOG):
                with open(CRASH_LOG, 'r') as f:
                    content = f.read()
                    
                    # Find the last generation section
                    gen_sections = re.findall(r'\[GEN (\d+)\]', content)
                    if gen_sections:
                        latest_gen = max([int(g) for g in gen_sections])
                        self.metrics["latest_gen"] = max(latest_gen, self.metrics["latest_gen"])
                        
                        # Find the crash types for this generation
                        latest_section = re.search(rf'\[GEN {latest_gen}\].*?(?=\[GEN|\Z)', content, re.DOTALL)
                        if latest_section:
                            section_text = latest_section.group(0)
                            
                            # Count different crash types
                            crash_types = {}
                            for line in section_text.splitlines():
                                if "Crash" in line:
                                    crash_type = "UNKNOWN"
                                    if "SIGNAL_" in line:
                                        # Extract signal type
                                        signal_match = re.search(r'SIGNAL_(\w+)', line)
                                        if signal_match:
                                            crash_type = f"SIGNAL_{signal_match.group(1)}"
                                    elif "exit code" in line:
                                        # Extract exit code
                                        exit_match = re.search(r'exit code (\d+)', line)
                                        if exit_match:
                                            crash_type = f"EXIT_{exit_match.group(1)}"
                                    elif "TIMEOUT" in line:
                                        crash_type = "TIMEOUT"
                                    elif "COMPILATION ERROR" in line:
                                        crash_type = "COMPILE_ERROR"
                                        
                                    if crash_type in crash_types:
                                        crash_types[crash_type] += 1
                                    else:
                                        crash_types[crash_type] = 1
                            
                            # Update or add to crash types
                            if crash_types:
                                self.metrics["crash_types"][str(latest_gen)] = crash_types
                            
                            # Count system impacts
                            system_impacts = section_text.count("[SYSTEM IMPACT]")
                            
                            # Update system impacts if we have enough generations
                            if latest_gen >= len(self.metrics["system_impacts"]):
                                while len(self.metrics["system_impacts"]) <= latest_gen:
                                    self.metrics["system_impacts"].append(0)
                                self.metrics["system_impacts"][latest_gen] = system_impacts
                            
                            # Count crashes
                            crash_count = section_text.count("Crash ")
                            total_count = len(re.findall(r'Shellcode:', section_text))
                            
                            # Update crash rate
                            if total_count > 0:
                                crash_rate = crash_count / total_count
                                
                                if latest_gen >= len(self.metrics["crash_rates"]):
                                    while len(self.metrics["crash_rates"]) <= latest_gen:
                                        self.metrics["crash_rates"].append(0)
                                    self.metrics["crash_rates"][latest_gen] = crash_rate
        except Exception as e:
            # Just ignore errors in this part
            pass
    
    def scan_survivor_log(self):
        """Scan the survivor log for latest population information"""
        try:
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r') as f:
                    content = f.read()
                    
                    # Find the last generation section
                    gen_sections = re.findall(r'\[GEN (\d+)\]', content)
                    if gen_sections:
                        latest_gen = max([int(g) for g in gen_sections])
                        self.metrics["latest_gen"] = max(latest_gen, self.metrics["latest_gen"])
                        
                        # Find the latest generation section
                        latest_section = re.search(rf'\[GEN {latest_gen}\].*?(?=\[GEN|\Z)', content, re.DOTALL)
                        if latest_section:
                            section_text = latest_section.group(0)
                            
                            # Count survivors
                            survivor_count = section_text.count("Survivor ")
                            self.metrics["population_size"] = survivor_count
                            
                            # Calculate average shellcode length
                            lengths = re.findall(r'len: (\d+)', section_text)
                            if lengths:
                                avg_length = sum([int(l) for l in lengths]) / len(lengths)
                                
                                if latest_gen >= len(self.metrics["shellcode_lengths"]):
                                    while len(self.metrics["shellcode_lengths"]) <= latest_gen:
                                        # Use the previous value as a placeholder if available
                                        prev_val = self.metrics["shellcode_lengths"][-1] if self.metrics["shellcode_lengths"] else 0
                                        self.metrics["shellcode_lengths"].append(prev_val)
                                    self.metrics["shellcode_lengths"][latest_gen] = avg_length
        except Exception as e:
            # Just ignore errors in this part
            pass
    
    def check_latest_generation(self):
        """Check directories to find latest generation"""
        try:
            if os.path.exists(OUTPUT_DIR):
                gen_dirs = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("gen_")]
                if gen_dirs:
                    # Extract generation numbers
                    gen_numbers = []
                    for d in gen_dirs:
                        match = re.search(r'gen_(\d+)', d)
                        if match:
                            gen_numbers.append(int(match.group(1)))
                    
                    if gen_numbers:
                        latest_gen = max(gen_numbers)
                        self.metrics["latest_gen"] = max(latest_gen, self.metrics["latest_gen"])
        except Exception as e:
            pass
    
    def analyze_latest_crash_types(self):
        """Analyze crash types from the latest generation"""
        latest_gen = self.metrics["latest_gen"]
        
        if str(latest_gen) in self.metrics["crash_types"]:
            crash_data = self.metrics["crash_types"][str(latest_gen)]
            return crash_data
        elif latest_gen in self.metrics["crash_types"]:
            crash_data = self.metrics["crash_types"][latest_gen]
            return crash_data
        
        return {}
    
    def get_system_health(self):
        """Estimate system health based on crash metrics"""
        try:
            # Check crash rates and system impacts
            if not self.metrics["crash_rates"]:
                return "Unknown"
            
            recent_crash_rates = self.metrics["crash_rates"][-5:]
            recent_system_impacts = self.metrics["system_impacts"][-5:]
            
            avg_crash_rate = sum(recent_crash_rates) / len(recent_crash_rates)
            total_recent_impacts = sum(recent_system_impacts)
            
            if avg_crash_rate > 0.8 and total_recent_impacts > 10:
                return "Critical - High Instability"
            elif avg_crash_rate > 0.5 and total_recent_impacts > 5:
                return "Warning - Moderate Instability"
            elif avg_crash_rate > 0.3 or total_recent_impacts > 2:
                return "Caution - Minor Instability"
            else:
                return "Good"
        except Exception:
            return "Unknown"
    
    def estimate_genetic_diversity(self):
        """Estimate genetic diversity from available data"""
        if self.reservoir_stats and "diversity_avg" in self.reservoir_stats:
            self.metrics["genetic_diversity"] = self.reservoir_stats["diversity_avg"]
        else:
            # If no reservoir stats, make a rough estimate based on shellcode lengths
            if len(self.metrics["shellcode_lengths"]) > 1:
                lengths = self.metrics["shellcode_lengths"]
                min_len, max_len = min(lengths), max(lengths)
                if max_len > 0:
                    normalized_range = (max_len - min_len) / max_len
                    self.metrics["genetic_diversity"] = normalized_range
    
    def update_metrics(self):
        """Update all metrics from all possible sources"""
        # First, clear any stale cached data
        self.metrics["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Load metrics from the metrics file - first priority
        self.load_metrics()
        
        # Try to load the genetic reservoir for diversity stats
        reservoir_loaded = self.load_reservoir()
        
        # Scan files to get additional information
        self.count_critical_crashes()
        self.estimate_population_size()
        
        # Check kernelhunter_crashes.txt for the latest crash info
        self.scan_crash_log()
        
        # Scan survivor log to get more accurate population data
        self.scan_survivor_log()
        
        # Calculate system health based on crash patterns
        self.metrics["system_health"] = self.get_system_health()
        
        # Estimate genetic diversity from available data
        self.estimate_genetic_diversity()
        
        # Force a refresh of latest generation count if needed
        self.check_latest_generation()
    
    def draw_mini_trend(self, stdscr, row, col, values, lower_is_better=True):
        """Draw a mini ASCII trend graph"""
        # Get color pairs
        GREEN = curses.color_pair(1)
        RED = curses.color_pair(2)
        
        if not values:
            stdscr.addstr(row, col, "No data")
            return
            
        # Prevent division by zero
        max_val = max(values) if values else 1
        
        # Normalize values between 0 and 1 if they aren't already
        if max_val > 1:
            normalized = [v/max_val for v in values]
        else:
            normalized = values
        
        # Using simple ASCII blocks for the trend
        trend_chars = " ▁▂▃▄▅▆▇█"
        trend_str = ""
        
        for val in normalized:
            # Convert value to index in trend_chars
            idx = int(val * (len(trend_chars) - 1))
            trend_str += trend_chars[idx]
        
        # Color based on trend direction
        if len(normalized) > 1:
            # Check if trend is improving or worsening
            start_val = normalized[0]
            end_val = normalized[-1]
            
            if lower_is_better:
                color = GREEN if end_val <= start_val else RED
            else:
                color = GREEN if end_val >= start_val else RED
        else:
            color = curses.color_pair(0)  # Default
        
        stdscr.addstr(row, col, trend_str, color)
    
    def format_runtime(self):
        """Format the runtime in a human-readable format"""
        runtime = time.time() - self.start_time
        
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"
    
    def draw_dashboard(self, stdscr):
        """Draw the main dashboard"""
        # Clear screen
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        # Colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Success
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)    # Error
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Warning
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Info
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # Special
        
        GREEN = curses.color_pair(1)
        RED = curses.color_pair(2)
        YELLOW = curses.color_pair(3)
        CYAN = curses.color_pair(4)
        MAGENTA = curses.color_pair(5)
        BOLD = curses.A_BOLD
        
        # Title
        title = "KernelHunter Evolution Monitor"
        stdscr.addstr(0, (width - len(title)) // 2, title, BOLD | CYAN)
        
        # Status bar
        status_line = f"Last Updated: {self.metrics['last_updated'] or 'Never'} | Run Time: {self.format_runtime()}"
        stdscr.addstr(height-1, 0, status_line, BOLD)
        
        # Instructions
        instruction_line = "Press 'q' to quit, 'r' to refresh"
        stdscr.addstr(height-1, width - len(instruction_line) - 1, instruction_line)
        
        # Draw main statistics
        
        # First column - Basic stats
        col1_x = 2
        row = 2
        
        stdscr.addstr(row, col1_x, "BASIC STATISTICS", BOLD | YELLOW)
        row += 1
        stdscr.addstr(row, col1_x, "=" * 30)
        row += 2
        
        # Generation info
        stdscr.addstr(row, col1_x, f"Latest Generation: ", BOLD)
        stdscr.addstr(f"{self.metrics['latest_gen']}")
        row += 1
        
        stdscr.addstr(row, col1_x, f"Population Size: ", BOLD)
        stdscr.addstr(f"{self.metrics['population_size']}")
        row += 1
        
        # Crash statistics
        if self.metrics["crash_rates"]:
            latest_crash_rate = self.metrics["crash_rates"][-1]
            crash_color = GREEN if latest_crash_rate < 0.3 else (YELLOW if latest_crash_rate < 0.7 else RED)
            
            stdscr.addstr(row, col1_x, f"Latest Crash Rate: ", BOLD)
            stdscr.addstr(f"{latest_crash_rate:.2%}", crash_color)
            row += 1
        
        # System impacts
        if self.metrics["system_impacts"]:
            latest_impact = self.metrics["system_impacts"][-1]
            impact_color = GREEN if latest_impact == 0 else (YELLOW if latest_impact < 5 else RED)
            
            stdscr.addstr(row, col1_x, f"System Impacts: ", BOLD)
            stdscr.addstr(f"{latest_impact} (last gen)", impact_color)
            row += 1
        
        # Critical crashes
        impact_color = GREEN if self.metrics["critical_crashes"] == 0 else (
            YELLOW if self.metrics["critical_crashes"] < 10 else RED)
        stdscr.addstr(row, col1_x, f"Critical Crashes: ", BOLD)
        stdscr.addstr(f"{self.metrics['critical_crashes']} total", impact_color)
        row += 1
        
        # Shellcode length
        if self.metrics["shellcode_lengths"]:
            latest_length = self.metrics["shellcode_lengths"][-1]
            stdscr.addstr(row, col1_x, f"Avg Shellcode Length: ", BOLD)
            stdscr.addstr(f"{latest_length:.1f} bytes")
            row += 1
        
        # System health
        health = self.metrics["system_health"]
        health_color = GREEN
        if health.startswith("Critical"):
            health_color = RED
        elif health.startswith("Warning"):
            health_color = YELLOW
        elif health.startswith("Caution"):
            health_color = YELLOW
        
        stdscr.addstr(row, col1_x, f"System Health: ", BOLD)
        stdscr.addstr(f"{health}", health_color)
        row += 2
        
        # Second column - Genetic diversity stats
        col2_x = width // 2
        row = 2
        
        stdscr.addstr(row, col2_x, "GENETIC DIVERSITY", BOLD | YELLOW)
        row += 1
        stdscr.addstr(row, col2_x, "=" * 30)
        row += 2
        
        # Reservoir stats
        stdscr.addstr(row, col2_x, f"Reservoir Size: ", BOLD)
        stdscr.addstr(f"{self.metrics['reservoir_size']} shellcodes")
        row += 1
        
        # Diversity measure
        stdscr.addstr(row, col2_x, f"Genetic Diversity: ", BOLD)
        diversity_val = self.metrics["genetic_diversity"]
        diversity_color = RED if diversity_val < 0.3 else (YELLOW if diversity_val < 0.6 else GREEN)
        stdscr.addstr(f"{diversity_val:.2f}", diversity_color)
        row += 1
        
        # Instruction diversity
        stdscr.addstr(row, col2_x, f"Instruction Types: ", BOLD)
        instr_count = len(self.metrics["instruction_diversity"])
        stdscr.addstr(f"{instr_count} different types")
        row += 2
        
        # Show top instruction types if available
        if self.metrics["instruction_diversity"]:
            stdscr.addstr(row, col2_x, "Top Instructions:", BOLD)
            row += 1
            
            # Sort by frequency
            sorted_instrs = sorted(self.metrics["instruction_diversity"].items(), 
                                  key=lambda x: x[1], reverse=True)
            
            # Show top 5
            for i, (instr_type, count) in enumerate(sorted_instrs[:5]):
                if row + i >= height - 3:
                    break
                stdscr.addstr(row + i, col2_x + 2, f"{instr_type}: {count}")
        row += 7  # Space for instruction types
        
        # Show trends section (first column continued)
        row_trends = 12
        stdscr.addstr(row_trends, col1_x, "EVOLUTION TRENDS", BOLD | YELLOW)
        row_trends += 1
        stdscr.addstr(row_trends, col1_x, "=" * 30)
        row_trends += 2
        
        # Draw mini trend graphs if we have data
        if len(self.metrics["crash_rates"]) > 1:
            # Crash rate trend (last 10 generations)
            stdscr.addstr(row_trends, col1_x, "Crash Rate Trend: ", BOLD)
            rates = self.metrics["crash_rates"][-10:]
            self.draw_mini_trend(stdscr, row_trends, col1_x + 18, rates, lower_is_better=True)
            row_trends += 1
        
        if len(self.metrics["shellcode_lengths"]) > 1:
            # Shellcode length trend
            stdscr.addstr(row_trends, col1_x, "Length Trend: ", BOLD)
            lengths = self.metrics["shellcode_lengths"][-10:]
            normalized_lengths = [l/max(lengths) for l in lengths] if max(lengths) > 0 else lengths
            self.draw_mini_trend(stdscr, row_trends, col1_x + 18, normalized_lengths, lower_is_better=False)
            row_trends += 1
        
        if len(self.metrics["system_impacts"]) > 1:
            # System impact trend
            stdscr.addstr(row_trends, col1_x, "Impact Trend: ", BOLD)
            impacts = self.metrics["system_impacts"][-10:]
            max_impact = max(impacts) if impacts else 1
            normalized_impacts = [i/max_impact for i in impacts] if max_impact > 0 else impacts
            self.draw_mini_trend(stdscr, row_trends, col1_x + 18, normalized_impacts, lower_is_better=True)
            row_trends += 2
        
        # Crash type distribution
        latest_crash_types = self.analyze_latest_crash_types()
        if latest_crash_types:
            stdscr.addstr(row_trends, col1_x, "Recent Crash Types:", BOLD)
            row_trends += 1
            
            # Sort by frequency
            sorted_types = sorted(latest_crash_types.items(), key=lambda x: x[1], reverse=True)
            
            # Show top 3
            for i, (crash_type, count) in enumerate(sorted_types[:3]):
                if row_trends + i >= height - 2:
                    break
                crash_name = crash_type[:20]  # Truncate if too long
                stdscr.addstr(row_trends + i, col1_x + 2, f"{crash_name}: {count}")
        
        # Refresh and update
        stdscr.refresh()
    
    def run(self, stdscr):
        """Main function to run the dashboard"""
        # Set up curses
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        stdscr.nodelay(True)  # Non-blocking input
        stdscr.timeout(200)   # Refresh every 200ms for more responsive UI
        
        # Initial metrics update
        self.update_metrics()
        
        # Main loop
        while True:
            # Auto-refresh every 1 second
            current_time = time.time()
            if current_time - self.last_refresh >= 1:
                self.update_metrics()
                self.last_refresh = current_time
                self.refresh_count += 1
            
            try:
                # Draw the dashboard
                self.draw_dashboard(stdscr)
                
                # Check for user input
                key = stdscr.getch()
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.update_metrics()
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                # Try to recover from any drawing errors
                stdscr.clear()
                stdscr.addstr(0, 0, f"Error: {str(e)}")
                stdscr.addstr(1, 0, "Press 'r' to retry or 'q' to quit")
                stdscr.refresh()
                
                # Wait for user input
                stdscr.nodelay(False)
                key = stdscr.getch()
                stdscr.nodelay(True)
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.update_metrics()


def main():
    """Entry point for the monitor"""
    monitor = KernelHunterMonitor()
    curses.wrapper(monitor.run)


if __name__ == "__main__":
    main()

