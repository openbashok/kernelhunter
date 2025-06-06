#!/usr/bin/env python3
import os
import time
import re
import curses
import csv
import json
from datetime import datetime
from collections import defaultdict, Counter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

KERN_LOG_PATH = "/var/log/kern.log"
CSV_LOG_PATH = "kernel_errors.csv"
FUZZER_CRASHES_DIR = "kernelhunter_crashes"
FUZZER_CRITICAL_DIR = "kernelhunter_critical"
METRICS_FILE = "kernelhunter_metrics.json"

# Kernel error patterns
KERNEL_ERROR_TYPES = {
    "BUG": re.compile(r"\bBUG:\s+(.*)", re.IGNORECASE),
    "Oops": re.compile(r"\bOops:\s+(.*)", re.IGNORECASE),
    "Segfault": re.compile(r"\bsegfault at\s+([0-9a-fx]+)", re.IGNORECASE),
    "General Protection Fault": re.compile(r"\bgeneral protection (ip|fault)", re.IGNORECASE),
    "Kernel Panic": re.compile(r"Kernel panic - not syncing: (.*)", re.IGNORECASE),
    "Bad Frame": re.compile(r"\bbad frame in\b", re.IGNORECASE),
    "Trap Alignment Check": re.compile(r"\btrap alignment check\b", re.IGNORECASE),
    "Trap Divide Error": re.compile(r"\btrap divide error\b", re.IGNORECASE),
    "Trap Int3": re.compile(r"\btrap int3 ip\b", re.IGNORECASE),
    "Trap Invalid Opcode": re.compile(r"\btrap invalid opcode\b", re.IGNORECASE),
    "Trap Overflow": re.compile(r"\btrap overflow ip\b", re.IGNORECASE),
    "Trap Stack Segment": re.compile(r"\btrap stack segment\b", re.IGNORECASE),
}

# Fuzzer signal types from KernelHunter with criticality levels
FUZZER_SIGNAL_TYPES = {
    # CRITICAL - Potential kernel exploitation/corruption
    "SIGNAL_SIGILL": {"level": 1, "desc": "Illegal Instruction"},
    "SIGNAL_SIGTRAP": {"level": 1, "desc": "Trap/Debug Signal"},
    "SIGNAL_SIGSYS": {"level": 1, "desc": "Bad System Call"},
    
    # HIGH - Memory corruption indicators
    "SIGNAL_SIGSEGV": {"level": 2, "desc": "Segmentation Fault"},
    "SIGNAL_SIGBUS": {"level": 2, "desc": "Bus Error"},
    "SIGNAL_SIGFPE": {"level": 2, "desc": "Floating Point Exception"},
    
    # MEDIUM - Process termination
    "SIGNAL_SIGABRT": {"level": 3, "desc": "Abort Signal"},
    "SIGNAL_SIGKILL": {"level": 3, "desc": "Kill Signal"},
    "SIGNAL_SIGTERM": {"level": 3, "desc": "Termination Signal"},
    "SIGNAL_SIGPIPE": {"level": 3, "desc": "Broken Pipe"},
    
    # LOW - Exit codes and timeouts
    "EXIT_1": {"level": 4, "desc": "Exit Code 1"},
    "EXIT_139": {"level": 4, "desc": "Exit Code 139 (SIGSEGV)"},
    "EXIT_127": {"level": 4, "desc": "Exit Code 127"},
    "EXIT_2": {"level": 4, "desc": "Exit Code 2"},
    "TIMEOUT": {"level": 5, "desc": "Execution Timeout"},
    "COMPILE_ERROR": {"level": 5, "desc": "Compilation Error"},
}

# Kernel error types with criticality levels
KERNEL_ERROR_CRITICALITY = {
    # CRITICAL - System compromise/corruption
    "Kernel Panic": {"level": 1, "desc": "Complete system failure"},
    "BUG": {"level": 1, "desc": "Kernel bug detected"},
    "General Protection Fault": {"level": 1, "desc": "Memory protection violation"},
    
    # HIGH - Memory/execution errors
    "Oops": {"level": 2, "desc": "Kernel oops - recoverable error"},
    "Segfault": {"level": 2, "desc": "Segmentation fault"},
    "Bad Frame": {"level": 2, "desc": "Stack frame corruption"},
    "Trap Invalid Opcode": {"level": 2, "desc": "Invalid CPU instruction"},
    
    # MEDIUM - CPU traps
    "Trap Divide Error": {"level": 3, "desc": "Division by zero"},
    "Trap Int3": {"level": 3, "desc": "Breakpoint trap"},
    "Trap Overflow": {"level": 3, "desc": "Arithmetic overflow"},
    "Trap Stack Segment": {"level": 3, "desc": "Stack segment fault"},
    "Trap Alignment Check": {"level": 3, "desc": "Memory alignment error"},
}

def load_existing_kernel_metrics():
    """Load existing kernel error metrics from CSV"""
    error_stats = defaultdict(int)
    
    # Initialize all known kernel error types to 0
    for error_type in KERNEL_ERROR_CRITICALITY.keys():
        error_stats[error_type] = 0
    
    if os.path.exists(CSV_LOG_PATH):
        try:
            with open(CSV_LOG_PATH, mode='r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 3:
                        error_type = row[2]
                        if error_type in KERNEL_ERROR_CRITICALITY:
                            error_stats[error_type] += 1
        except Exception as e:
            print(f"Error loading kernel metrics: {e}")
    
    return error_stats

def load_fuzzer_signal_metrics():
    """Load fuzzer signal metrics from crash JSON files"""
    fuzzer_stats = defaultdict(int)
    
    # Initialize all known signal types to 0
    for signal_type in FUZZER_SIGNAL_TYPES.keys():
        fuzzer_stats[signal_type] = 0
    
    # Check both crash directories
    for crash_dir in [FUZZER_CRASHES_DIR, FUZZER_CRITICAL_DIR]:
        if not os.path.exists(crash_dir):
            continue
            
        try:
            for filename in os.listdir(crash_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(crash_dir, filename)
                    
                    try:
                        with open(filepath, 'r') as f:
                            crash_data = json.load(f)
                            
                        crash_type = crash_data.get('crash_type', 'UNKNOWN')
                        if crash_type in FUZZER_SIGNAL_TYPES or crash_type.startswith(('SIGNAL_', 'EXIT_')):
                            fuzzer_stats[crash_type] += 1
                            
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
        except Exception as e:
            print(f"Error loading fuzzer metrics from {crash_dir}: {e}")
    
    return fuzzer_stats

def load_realtime_fuzzer_metrics():
    """Load real-time metrics from KernelHunter metrics.json"""
    realtime_stats = defaultdict(int)
    
    # Initialize all known signal types to 0
    for signal_type in FUZZER_SIGNAL_TYPES.keys():
        realtime_stats[signal_type] = 0
    
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                metrics = json.load(f)
            
            # Get crash types from all generations
            crash_types = metrics.get('crash_types', {})
            for gen_data in crash_types.values():
                if isinstance(gen_data, dict):
                    for crash_type, count in gen_data.items():
                        if crash_type in FUZZER_SIGNAL_TYPES:
                            realtime_stats[crash_type] += count
                        
        except Exception as e:
            print(f"Error loading real-time metrics: {e}")
    
    return realtime_stats

def get_criticality_color(level):
    """Get color based on criticality level"""
    color_map = {
        1: curses.color_pair(1),  # RED - Critical
        2: curses.color_pair(6),  # BRIGHT RED - High  
        3: curses.color_pair(2),  # YELLOW - Medium
        4: curses.color_pair(3),  # GREEN - Low
        5: curses.color_pair(7),  # WHITE - Very Low
    }
    return color_map.get(level, curses.color_pair(3))

def sort_by_criticality(error_dict, error_types_def):
    """Sort errors by criticality level, then by count"""
    def get_sort_key(item):
        error_type, count = item
        level = error_types_def.get(error_type, {}).get('level', 99)
        return (level, -count)  # Sort by level ASC, then count DESC
    
    return sorted(error_dict.items(), key=get_sort_key)

class EnhancedKernelLogHandler(FileSystemEventHandler):
    def __init__(self, logfile, kernel_error_stats, fuzzer_signal_stats):
        self.logfile = logfile
        self.inode = os.stat(logfile).st_ino if os.path.exists(logfile) else None
        self.file = None
        if os.path.exists(logfile):
            self.file = open(logfile, 'r')
            self.file.seek(0, 2)  # Go to end
        
        self.kernel_error_stats = kernel_error_stats
        self.fuzzer_signal_stats = fuzzer_signal_stats

        if not os.path.exists(CSV_LOG_PATH):
            with open(CSV_LOG_PATH, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Timestamp", "Program", "Error Type", "Message", "Address", "Original Line"])

    def on_modified(self, event):
        if not self.file or not os.path.exists(self.logfile):
            return
            
        try:
            current_inode = os.stat(self.logfile).st_ino
            if current_inode != self.inode:
                if self.file:
                    self.file.close()
                time.sleep(0.5)
                self.file = open(self.logfile, 'r')
                self.inode = current_inode
        except:
            return

        self.parse_new_lines()

    def parse_new_lines(self):
        if not self.file:
            return
            
        for line in self.file:
            for error_type, pattern in KERNEL_ERROR_TYPES.items():
                match = pattern.search(line)
                if match:
                    self.kernel_error_stats[error_type] += 1
                    self.log_to_csv(line, error_type, match)
                    break

    def log_to_csv(self, line, error_type, match):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        program_name = self.extract_program_name(line)
        address = match.group(1) if match.groups() else "N/A"
        message = match.group(0)

        with open(CSV_LOG_PATH, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, program_name, error_type, message, address, line.strip()])

    def extract_program_name(self, line):
        match = re.search(r"\b(\S+)\[\d+\]", line)
        return match.group(1) if match else "Unknown"

    def update_fuzzer_stats(self):
        """Update fuzzer signal statistics from all sources"""
        # Combine metrics from JSON files and real-time metrics
        file_stats = load_fuzzer_signal_metrics()
        realtime_stats = load_realtime_fuzzer_metrics()
        
        # Merge both sources (realtime takes precedence for overlaps)
        combined_stats = file_stats.copy()
        for signal_type, count in realtime_stats.items():
            if count > combined_stats.get(signal_type, 0):
                combined_stats[signal_type] = count
        
        # Update the instance stats
        self.fuzzer_signal_stats.clear()
        self.fuzzer_signal_stats.update(combined_stats)

def draw_enhanced_dashboard(stdscr, kernel_error_stats, fuzzer_signal_stats, handler):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(500)
    
    last_update = time.time()
    update_interval = 2  # Update every 2 seconds

    # Initialize colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)        # Critical
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)     # Medium
    curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)      # Low
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)       # Header
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)    # Fuzzer
    curses.init_pair(6, curses.COLOR_RED, curses.COLOR_BLACK)        # High (bright red)
    curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)      # Very Low

    RED = curses.color_pair(1)
    YELLOW = curses.color_pair(2)
    GREEN = curses.color_pair(3)
    CYAN = curses.color_pair(4)
    MAGENTA = curses.color_pair(5)
    BRIGHT_RED = curses.color_pair(6)
    WHITE = curses.color_pair(7)

    while True:
        current_time = time.time()
        
        # Auto-update fuzzer stats
        if current_time - last_update >= update_interval:
            handler.update_fuzzer_stats()
            last_update = current_time
        
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Title with last update time
        title = "Enhanced Kernel Error Monitor (KernelHunter + Kernel Logs)"
        update_time = time.strftime("%H:%M:%S")
        title_with_time = f"{title} - Updated: {update_time}"
        stdscr.addstr(1, max(0, (width // 2) - len(title_with_time) // 2), title_with_time, curses.A_BOLD | CYAN)

        # Two-column layout
        col1_width = width // 2 - 2
        col2_start = width // 2 + 1

        # Left column: Kernel Errors
        stdscr.addstr(3, 2, "KERNEL LOG ERRORS (by Criticality)", curses.A_BOLD | RED)
        stdscr.addstr(4, 2, "=" * (col1_width - 1))

        row = 6
        total_kernel_errors = sum(kernel_error_stats.values())
        
        # Sort kernel errors by criticality
        sorted_kernel = sort_by_criticality(kernel_error_stats, KERNEL_ERROR_CRITICALITY)
        
        for error_type, count in sorted_kernel:
            if row >= height - 5:
                break
                
            # Get criticality info
            error_info = KERNEL_ERROR_CRITICALITY.get(error_type, {"level": 99, "desc": "Unknown"})
            level = error_info["level"]
            desc = error_info["desc"]
            
            # Get color based on criticality
            color = get_criticality_color(level)
            
            # Format line with criticality indicator
            criticality_indicator = "🔥" if level == 1 else "⚠️" if level == 2 else "⚡" if level == 3 else "ℹ️"
            line = f"{criticality_indicator} {error_type[:col1_width-15]} | {count}"
            
            # Add attribute if count > 0
            attr = curses.A_BOLD if count > 0 else curses.A_DIM
            
            stdscr.addstr(row, 2, line[:col1_width-1], color | attr)
            row += 1

        # Right column: Fuzzer Signals  
        stdscr.addstr(3, col2_start, "FUZZER DETECTED SIGNALS (by Criticality)", curses.A_BOLD | MAGENTA)
        stdscr.addstr(4, col2_start, "=" * (col1_width - 1))

        row = 6
        total_fuzzer_signals = sum(handler.fuzzer_signal_stats.values())
        
        # Sort fuzzer signals by criticality
        sorted_fuzzer = sort_by_criticality(handler.fuzzer_signal_stats, FUZZER_SIGNAL_TYPES)
        
        for signal_type, count in sorted_fuzzer:
            if row >= height - 5:
                break
            
            # Get criticality info
            signal_info = FUZZER_SIGNAL_TYPES.get(signal_type, {"level": 99, "desc": "Unknown"})
            level = signal_info["level"]
            desc = signal_info["desc"]
            
            # Get color based on criticality
            color = get_criticality_color(level)
            
            # Format line with criticality indicator
            criticality_indicator = "🔥" if level == 1 else "⚠️" if level == 2 else "⚡" if level == 3 else "ℹ️" if level == 4 else "💤"
            line = f"{criticality_indicator} {signal_type[:col1_width-15]} | {count}"
            
            # Add attribute if count > 0
            attr = curses.A_BOLD if count > 0 else curses.A_DIM
            
            stdscr.addstr(row, col2_start, line[:col1_width-1], color | attr)
            row += 1

        # Summary stats at bottom
        summary_row = height - 4
        stdscr.addstr(summary_row, 2, 
                     f"Total Kernel Errors: {total_kernel_errors} | Total Fuzzer Signals: {total_fuzzer_signals}", 
                     curses.A_BOLD)
        
        # Legend
        legend_row = height - 3
        stdscr.addstr(legend_row, 2, "🔥=Critical  ⚠️=High  ⚡=Medium  ℹ️=Low  💤=VeryLow", curses.A_DIM)
        
        # Instructions
        stdscr.addstr(height - 2, 2, "Press 'r' to refresh | 'q' to quit | Auto-updates every 2s | BOLD=Active")
        stdscr.addstr(height - 1, 2, "Order: Criticality first, then by count | DIM=Zero occurrences", curses.A_DIM)

        stdscr.refresh()

        # Handle input
        try:
            key = stdscr.getch()
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Force immediate refresh
                handler.update_fuzzer_stats()
                last_update = current_time
        except KeyboardInterrupt:
            break

def start_enhanced_dashboard():
    # Load existing metrics
    kernel_error_stats = load_existing_kernel_metrics()
    
    # Set up file monitoring for kernel log and fuzzer stats
    handler = EnhancedKernelLogHandler(KERN_LOG_PATH, kernel_error_stats, defaultdict(int))
    
    # Initialize fuzzer stats
    handler.update_fuzzer_stats()
    
    observer = Observer()
    
    if os.path.exists(os.path.dirname(KERN_LOG_PATH)):
        observer.schedule(handler, path=os.path.dirname(KERN_LOG_PATH), recursive=False)
        observer.start()

    # Also monitor the fuzzer directories for changes
    fuzzer_observer = Observer()
    for directory in [FUZZER_CRASHES_DIR, FUZZER_CRITICAL_DIR, os.path.dirname(METRICS_FILE)]:
        if os.path.exists(directory):
            fuzzer_observer.schedule(handler, path=directory, recursive=False)
    fuzzer_observer.start()

    try:
        curses.wrapper(draw_enhanced_dashboard, kernel_error_stats, handler.fuzzer_signal_stats, handler)
                
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        fuzzer_observer.stop()
        fuzzer_observer.join()

if __name__ == "__main__":
    start_enhanced_dashboard()
