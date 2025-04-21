import os
import time
import re
import curses
import csv
from datetime import datetime
from collections import defaultdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

KERN_LOG_PATH = "/var/log/kern.log"
CSV_LOG_PATH = "kernel_errors.csv"

ERROR_TYPES = {
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

def load_existing_metrics():
    """Load existing metrics from the CSV file"""
    error_stats = defaultdict(int)
    
    if os.path.exists(CSV_LOG_PATH):
        try:
            with open(CSV_LOG_PATH, mode='r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Saltar la fila de encabezados
                
                for row in reader:
                    if len(row) >= 3:  # Asegurarse de que hay suficientes columnas
                        error_type = row[2]  # La tercera columna contiene el tipo de error
                        if error_type in ERROR_TYPES:
                            error_stats[error_type] += 1
        except Exception as e:
            print(f"Error al cargar métricas existentes: {e}")
    
    return error_stats

class KernelLogHandler(FileSystemEventHandler):
    def __init__(self, logfile, error_stats):
        self.logfile = logfile
        self.inode = os.stat(logfile).st_ino
        self.file = open(logfile, 'r')
        self.file.seek(0, 2)
        self.error_stats = error_stats

        if not os.path.exists(CSV_LOG_PATH):
            with open(CSV_LOG_PATH, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Timestamp", "Program", "Error Type", "Message", "Address", "Original Line"])

    def on_modified(self, event):
        if not os.path.exists(self.logfile) or os.stat(self.logfile).st_ino != self.inode:
            self.file.close()
            time.sleep(0.5)
            self.file = open(self.logfile, 'r')
            self.inode = os.stat(self.logfile).st_ino

        self.parse_new_lines()

    def parse_new_lines(self):
        for line in self.file:
            for error_type, pattern in ERROR_TYPES.items():
                match = pattern.search(line)
                if match:
                    self.error_stats[error_type] += 1
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

def draw_dashboard(stdscr, error_stats):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(500)

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        title = "Kernel Error Monitor (kernelhunter)"
        stdscr.addstr(1, (width // 2) - len(title) // 2, title, curses.A_BOLD)

        stdscr.addstr(3, 4, "Error Type".ljust(30) + "| Occurrences", curses.A_UNDERLINE)

        row = 5
        for error_type in ERROR_TYPES:
            count = error_stats.get(error_type, 0)
            stdscr.addstr(row, 4, error_type.ljust(30) + f"| {count}")
            row += 1

        stdscr.addstr(row + 2, 4, "(Press CTRL+C to exit)")

        stdscr.refresh()

        try:
            key = stdscr.getch()
            if key == ord('q'):
                break
        except KeyboardInterrupt:
            break

def start_dashboard():
    # Cargar métricas existentes desde el CSV
    error_stats = load_existing_metrics()
    
    handler = KernelLogHandler(KERN_LOG_PATH, error_stats)
    observer = Observer()
    observer.schedule(handler, path=os.path.dirname(KERN_LOG_PATH), recursive=False)
    observer.start()

    try:
        curses.wrapper(draw_dashboard, error_stats)
    finally:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    start_dashboard()