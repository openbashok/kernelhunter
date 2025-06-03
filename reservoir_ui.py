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
        sort_key = "index"  # index, length or preview
        reverse = False
        while True:
            stdscr.clear()
            stdscr.addstr(0, 2, "Shellcodes (q to return)", curses.A_BOLD)
            stdscr.addstr(1, 2, f"Sort: {sort_key}{' desc' if reverse else ' asc'}  [s] cycle  [r] reverse")

            sorted_list = list(enumerate(self.reservoir.reservoir))
            if sort_key == "length":
                sorted_list.sort(key=lambda x: len(x[1]), reverse=reverse)
            elif sort_key == "preview":
                sorted_list.sort(key=lambda x: x[1][:16].hex(), reverse=reverse)
            else:
                sorted_list.sort(key=lambda x: x[0], reverse=reverse)

            height, width = stdscr.getmaxyx()
            visible = sorted_list[offset:offset + height - 5]
            for i, (orig_idx, sc) in enumerate(visible):
                preview = " ".join(f"{b:02x}" for b in sc[:16])
                line = f"{orig_idx:04d} | len {len(sc):3d} | {preview}"
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
            elif key == curses.KEY_DOWN and idx < len(sorted_list) - 1:
                idx += 1
                if idx >= offset + (height - 5):
                    offset += 1
            elif key == ord('s'):
                sort_key = {'index': 'length', 'length': 'preview', 'preview': 'index'}[sort_key]
                idx, offset = 0, 0
            elif key == ord('r'):
                reverse = not reverse
            elif key == ord('d') and sorted_list:
                self.delete_shellcode(stdscr, sorted_list[idx][0])
                if idx >= len(self.reservoir):
                    idx = max(0, len(self.reservoir) - 1)
                if offset > idx:
                    offset = idx
            elif key == ord('e') and sorted_list:
                self.edit_shellcode(stdscr, sorted_list[idx][0])
            elif key == ord('a') and sorted_list:
                self.analyze_shellcode(stdscr, sorted_list[idx][0])

    def delete_shellcode(self, stdscr, orig_idx):
        stdscr.clear()
        stdscr.addstr(0, 2, f"Delete shellcode {orig_idx}? (y/n)")
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord('y'), ord('Y')):
            try:
                self.reservoir.reservoir.pop(orig_idx)
                self.reservoir.clear_cache()
                self.save()
                self.show_message(stdscr, "Shellcode deleted")
            except Exception as e:
                self.show_message(stdscr, f"Error: {e}")
        else:
            self.show_message(stdscr, "Cancelled")

    def edit_shellcode(self, stdscr, orig_idx):
        from crispr_mutation import crispr_edit_shellcode

        shellcode = self.reservoir.reservoir[orig_idx]
        edited = crispr_edit_shellcode(shellcode)
        self.reservoir.reservoir[orig_idx] = edited
        self.reservoir.clear_cache()
        self.reservoir.extract_features(edited)
        self.save()
        self.show_message(stdscr, "Shellcode edited")

    def analyze_shellcode(self, stdscr, orig_idx):
        shellcode = self.reservoir.reservoir[orig_idx]
        features = self.reservoir.extract_features(shellcode)
        stdscr.clear()
        stdscr.addstr(0, 2, f"Shellcode {orig_idx} analysis", curses.A_BOLD)
        stdscr.addstr(2, 2, f"Length: {features.get('length', 0)}")
        stdscr.addstr(3, 2, f"Syscalls: {features.get('syscalls', 0)}")
        stdscr.addstr(4, 2, f"Privileged instr: {features.get('privileged_instr', 0)}")
        stdscr.addstr(6, 2, "Instruction types:")
        types = features.get('instruction_types', {})
        line = 7
        for name, count in types.items():
            stdscr.addstr(line, 4, f"{name}: {count}")
            line += 1
        preview = ' '.join(f"{b:02x}" for b in shellcode[:32])
        stdscr.addstr(line + 1, 2, preview)
        stdscr.addstr(line + 3, 2, "Press any key to return")
        stdscr.refresh()
        stdscr.getch()

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
