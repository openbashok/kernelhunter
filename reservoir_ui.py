#!/usr/bin/env python3
"""Interactive TUI to inspect KernelHunter's genetic reservoir."""
import curses
import os
import time
from genetic_reservoir import GeneticReservoir
try:
    from kernelhunter_config import get_reservoir_file
except Exception:
    def get_reservoir_file(name="kernelhunter_reservoir.pkl"):
        return name

# Store the reservoir next to this script so it is consistent regardless of the
# current working directory.
#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#RESERVOIR_FILE = os.path.join(SCRIPT_DIR, "kernelhunter_reservoir.pkl")
RESERVOIR_FILE = get_reservoir_file()


class ReservoirUI:
    def __init__(self, refresh_seconds: int = 5, loop_delay_ms: int = 200):
        self.reservoir = GeneticReservoir()
        self.refresh_seconds = refresh_seconds
        self.loop_delay_ms = loop_delay_ms
        self.last_reload = time.time()
        self.load()

    def load(self):
        if os.path.exists(RESERVOIR_FILE):
            return self.reservoir.load_from_file(RESERVOIR_FILE)
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
            stdscr.timeout(-1)
            stdscr.getch()
            stdscr.timeout(self.loop_delay_ms)
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
        stdscr.timeout(-1)
        stdscr.getch()
        stdscr.timeout(self.loop_delay_ms)

    def list_shellcodes(self, stdscr):
        idx = 0
        offset = 0
        sort_key = "index"
        ascending = True
        stdscr.timeout(self.loop_delay_ms)
        last_reload = time.time()
        while True:
            stdscr.clear()
            stdscr.addstr(0, 2, "Shellcodes (q to return)", curses.A_BOLD)
            stdscr.addstr(1, 2, "Arrows: move  d:delete  e:edit  a:analyze  s:sort", curses.A_DIM)
            height, width = stdscr.getmaxyx()

            now = time.time()
            if now - last_reload >= self.refresh_seconds:
                self.load()
                last_reload = now

            enumerated = list(enumerate(self.reservoir.reservoir))
            if sort_key == "length":
                enumerated.sort(key=lambda x: len(x[1]), reverse=not ascending)
            elif sort_key == "index" and not ascending:
                enumerated.reverse()

            visible = enumerated[offset:offset + height - 5]
            for i, (real_idx, sc) in enumerate(visible):
                prefix = f"{real_idx:04d} | {len(sc):4d} | "
                avail_cols = width - 4 - len(prefix)
                if avail_cols > 0:
                    max_bytes = min(len(sc), (avail_cols + 1) // 3)
                else:
                    max_bytes = 0
                preview = " ".join(f"{b:02x}" for b in sc[:max_bytes])
                line = f"{prefix}{preview}"
                attr = curses.A_REVERSE if offset + i == idx else curses.A_NORMAL
                stdscr.addstr(i + 2, 2, line[:width - 4], attr)

            stdscr.refresh()
            key = stdscr.getch()
            if key == -1:
                continue
            if key in (ord('q'), 27):
                break
            if key == curses.KEY_UP and idx > 0:
                idx -= 1
                if idx < offset:
                    offset -= 1
            elif key == curses.KEY_DOWN and idx < len(enumerated) - 1:
                idx += 1
                if idx >= offset + (height - 5):
                    offset += 1
            elif key == ord('d'):
                if self.reservoir.remove(enumerated[idx][0]):
                    if idx >= len(enumerated) - 1 and idx > 0:
                        idx -= 1
                    self.save()
            elif key == ord('e'):
                self.edit_shellcode(stdscr, enumerated[idx][0])
            elif key == ord('a'):
                self.analyze_shellcode(stdscr, enumerated[idx][0])
            elif key == ord('s'):
                if sort_key == "index":
                    sort_key = "length"
                elif sort_key == "length" and ascending:
                    ascending = False
                else:
                    sort_key = "index"
                    ascending = True

    def edit_shellcode(self, stdscr, index):
        """Open an external editor to modify the shellcode."""
        import tempfile
        import subprocess

        sc = self.reservoir.reservoir[index]
        editor = os.environ.get("EDITOR", "nano")

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp.write(sc.hex())
            tmp_path = tmp.name

        # Suspend curses, open editor, then resume
        curses.def_prog_mode()
        curses.endwin()
        subprocess.call([editor, tmp_path])
        curses.reset_prog_mode()
        curses.curs_set(0)
        stdscr.clear()
        stdscr.refresh()

        try:
            with open(tmp_path, "r") as f:
                new_hex = f.read().strip()
            os.remove(tmp_path)
            new_bytes = bytes.fromhex(new_hex)
        except Exception:
            self.show_message(stdscr, "Invalid hex input")
            return

        if self.reservoir.update(index, new_bytes):
            self.save()
            self.show_message(stdscr, "Shellcode updated")
        else:
            self.show_message(stdscr, "Update failed")

    def _generate_analysis_report(self, shellcode):
        """Return a list of textual lines summarising the shellcode."""
        from collections import Counter
        import math
        try:
            from kernelhunter import interpret_instruction
        except Exception:  # fallback when running from source
            try:
                from kernelHunter import interpret_instruction
            except Exception:
                def interpret_instruction(_bytes):
                    return "instrucción desconocida"

        features = self.reservoir.extract_features(shellcode)

        lines = []
        lines.append(f"Length: {features.get('length', len(shellcode))} bytes")
        lines.append(f"Syscalls: {features.get('syscalls', 0)}")
        lines.append(f"Privileged instructions: {features.get('privileged_instr', 0)}")

        instr_types = features.get('instruction_types', {})
        if instr_types:
            lines.append("Instruction type counts:")
            for t_name, count in instr_types.items():
                lines.append(f"  {t_name}: {count}")

        # Byte distribution metrics
        counter = Counter(shellcode)
        total = len(shellcode)
        entropy = 0.0
        for c in counter.values():
            p = c / total
            entropy -= p * math.log2(p)

        lines.append(f"Unique bytes: {len(counter)}")
        lines.append(f"Shannon entropy: {entropy:.2f} bits/byte")

        lines.append("Top 5 bytes:")
        for b, cnt in counter.most_common(5):
            lines.append(f"  0x{b:02x}: {cnt}")

        lines.append("Byte occurrences:")
        for b in sorted(counter):
            lines.append(f"  0x{b:02x}: {counter[b]}")

        # Detect known instruction patterns
        from collections import Counter

        found = Counter()
        for i in range(len(shellcode)):
            desc = interpret_instruction(shellcode[i:i+8])
            if desc != "instrucción desconocida":
                found[desc] += 1

        lines.append("Known instructions detected:")
        if found:
            for d, cnt in found.most_common():
                lines.append(f"  - {d}")
        else:
            lines.append("  None")

        if found:
            lines.append("Instruction occurrences:")
            for d, cnt in found.most_common():
                lines.append(f"  {d}: {cnt}")

        return lines

    def analyze_shellcode(self, stdscr, index):
        """Display a detailed textual analysis for a shellcode."""
        import tempfile
        import subprocess

        shellcode = self.reservoir.reservoir[index]
        report = self._generate_analysis_report(shellcode)

        editor = os.environ.get("EDITOR", "nano")
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp.write("\n".join(report))
            tmp_path = tmp.name

        curses.def_prog_mode()
        curses.endwin()
        subprocess.call([editor, tmp_path])
        curses.reset_prog_mode()
        curses.curs_set(0)
        stdscr.clear()
        stdscr.refresh()

        try:
            os.remove(tmp_path)
        except OSError:
            pass

        stdscr.addstr(0, 2, "Press any key to return")
        stdscr.refresh()
        stdscr.timeout(-1)
        stdscr.getch()
        stdscr.timeout(self.loop_delay_ms)

    def export_reservoir(self, stdscr):
        curses.echo()
        stdscr.clear()
        stdscr.addstr(0, 2, "Export filename: ")
        stdscr.refresh()
        stdscr.timeout(-1)
        path = stdscr.getstr().decode().strip()
        stdscr.timeout(self.loop_delay_ms)
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
        stdscr.timeout(-1)
        key = stdscr.getch()
        stdscr.timeout(self.loop_delay_ms)
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
        stdscr.timeout(-1)
        stdscr.getch()
        stdscr.timeout(self.loop_delay_ms)

    # -------------------- Main loop --------------------
    def run(self, stdscr):
        curses.curs_set(0)
        stdscr.timeout(self.loop_delay_ms)
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
            now = time.time()
            if now - self.last_reload >= self.refresh_seconds:
                self.load()
                self.last_reload = now

            stdscr.clear()
            stdscr.addstr(0, 2, "KernelHunter - Reservoir Manager", curses.A_BOLD)
            for i, opt in enumerate(options):
                attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
                stdscr.addstr(i + 2, 4, opt, attr)
            stdscr.refresh()
            key = stdscr.getch()
            if key == -1:
                continue
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
