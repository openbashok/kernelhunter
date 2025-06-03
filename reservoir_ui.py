#!/usr/bin/env python3
"""Interactive TUI to inspect KernelHunter's genetic reservoir."""
import curses
import os
from genetic_reservoir import GeneticReservoir

RESERVOIR_FILE = "kernelhunter_reservoir.pkl"

class ReservoirUI:
    def __init__(self):
        self.reservoir = GeneticReservoir()
        self.load()

    def load(self):
        if os.path.exists(RESERVOIR_FILE) and self.reservoir.load_from_file(RESERVOIR_FILE):
            return True
        self.reservoir = GeneticReservoir()
        return False

    def save(self):
        self.reservoir.save_to_file(RESERVOIR_FILE)

    # -------------------- UI helpers --------------------
    def show_message(self, stdscr, message, pause=True):
        stdscr.clear()
        stdscr.addstr(0, 2, message)
        if pause:
            stdscr.addstr(2, 2, "Press any key to continue")
            stdscr.refresh()
            stdscr.getch()
        else:
            stdscr.refresh()

    # -------------------- Menu actions --------------------
    def show_summary(self, stdscr):
        stats = self.reservoir.get_diversity_stats()
        lines = [
            f"Reservoir size: {len(self.reservoir)}",
            f"Unique crash types: {stats.get('unique_crash_types', 0)}",
            f"Average diversity: {stats.get('diversity_avg', 0):.3f}",
            f"Min diversity: {stats.get('diversity_min', 0):.3f}",
            f"Max diversity: {stats.get('diversity_max', 0):.3f}",
            f"Average shellcode length: {stats.get('avg_shellcode_length', 0):.1f}",
        ]
        stdscr.clear()
        stdscr.addstr(0, 2, "Reservoir Summary", curses.A_BOLD)
        for i, line in enumerate(lines):
            stdscr.addstr(i + 2, 2, line)
        stdscr.addstr(len(lines) + 3, 2, "Press any key to return")
        stdscr.refresh()
        stdscr.getch()

    def list_shellcodes(self, stdscr):
        idx = 0
        offset = 0
        while True:
            stdscr.clear()
            stdscr.addstr(0, 2, "Shellcodes (q to return)", curses.A_BOLD)
            height, width = stdscr.getmaxyx()
            visible = self.reservoir.reservoir[offset:offset + height - 4]
            for i, sc in enumerate(visible):
                preview = " ".join(f"{b:02x}" for b in sc[:8])
                line = f"{offset + i:04d}: len {len(sc):3d} | {preview}"
                attr = curses.A_REVERSE if offset + i == idx else curses.A_NORMAL
                stdscr.addstr(i + 2, 2, line[:width - 4], attr)
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord('q'), 27):
                break
            if key == curses.KEY_UP and idx > 0:
                idx -= 1
                if idx < offset:
                    offset -= 1
            elif key == curses.KEY_DOWN and idx < len(self.reservoir) - 1:
                idx += 1
                if idx >= offset + (height - 4):
                    offset += 1

    def export_reservoir(self, stdscr):
        curses.echo()
        stdscr.clear()
        stdscr.addstr(0, 2, "Export filename: ")
        stdscr.refresh()
        path = stdscr.getstr().decode().strip()
        curses.noecho()
        if path:
            try:
                self.reservoir.save_to_file(path)
                self.show_message(stdscr, f"Saved to {path}")
            except Exception as e:
                self.show_message(stdscr, f"Error: {e}")
        else:
            self.show_message(stdscr, "Export cancelled")

    def clear_reservoir(self, stdscr):
        stdscr.clear()
        stdscr.addstr(0, 2, "Clear reservoir and delete file? (y/n)")
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord('y'), ord('Y')):
            self.reservoir = GeneticReservoir()
            if os.path.exists(RESERVOIR_FILE):
                try:
                    os.remove(RESERVOIR_FILE)
                except OSError:
                    pass
            self.show_message(stdscr, "Reservoir cleared")
        else:
            self.show_message(stdscr, "Cancelled")

    def reload_reservoir(self, stdscr):
        if self.load():
            self.show_message(stdscr, "Reservoir loaded", pause=False)
        else:
            self.show_message(stdscr, "Reservoir file not found; new reservoir created", pause=False)
        stdscr.getch()

    # -------------------- Main loop --------------------
    def run(self, stdscr):
        curses.curs_set(0)
        options = [
            "View summary",
            "List shellcodes",
            "Export reservoir",
            "Clear reservoir",
            "Reload reservoir",
            "Quit",
        ]
        idx = 0
        while True:
            stdscr.clear()
            stdscr.addstr(0, 2, "KernelHunter - Reservoir Manager", curses.A_BOLD)
            for i, opt in enumerate(options):
                attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                stdscr.addstr(i + 2, 4, opt, attr)
            stdscr.refresh()
            key = stdscr.getch()
            if key == curses.KEY_UP and idx > 0:
                idx -= 1
            elif key == curses.KEY_DOWN and idx < len(options) - 1:
                idx += 1
            elif key in (curses.KEY_ENTER, ord('\n')):
                sel = options[idx]
                if sel == "View summary":
                    self.show_summary(stdscr)
                elif sel == "List shellcodes":
                    self.list_shellcodes(stdscr)
                elif sel == "Export reservoir":
                    self.export_reservoir(stdscr)
                elif sel == "Clear reservoir":
                    self.clear_reservoir(stdscr)
                elif sel == "Reload reservoir":
                    self.reload_reservoir(stdscr)
                elif sel == "Quit":
                    break
            elif key in (ord('q'), 27):
                break


def main():
    ui = ReservoirUI()
    curses.wrapper(ui.run)


if __name__ == "__main__":
    main()
