#!/usr/bin/env python3
import curses
import json
import os
import time
import math

METRICS_FILE = "kernelhunter_metrics.json"

class AttackMutationMonitor:
    def __init__(self):
        self.metrics = {
            "attack_stats": {},
            "mutation_stats": {},
            "attack_totals": {},
            "mutation_totals": {},
            "latest_gen": 0,
        }
        self.attack_totals = {}
        self.mutation_totals = {}
        self.last_refresh = 0

    def _draw_items_in_columns(self, stdscr, row, col, items, available_width, max_rows):
        """Helper to draw key/value pairs in columns to better use the screen."""
        if not items:
            return

        # Prepare formatted strings
        formatted = [f"{name}: {count}" for name, count in items]
        if not formatted:
            return

        # Determine optimal column width
        max_len = max(len(s) for s in formatted) + 2
        num_cols = max(1, available_width // max_len)
        rows_needed = math.ceil(len(formatted) / num_cols)

        if rows_needed > max_rows and max_rows > 0:
            num_cols = math.ceil(len(formatted) / max_rows)
            rows_needed = math.ceil(len(formatted) / num_cols)
            max_len = max(1, available_width // num_cols)

        for idx, text in enumerate(formatted):
            r = row + (idx % rows_needed)
            c = col + (idx // rows_needed) * max_len
            if r >= row + max_rows:
                break
            stdscr.addstr(r, c, text[:max_len - 1])

    def load_metrics(self):
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, 'r') as f:
                    data = json.load(f)
                self.metrics["attack_stats"] = data.get("attack_stats", {})
                self.metrics["mutation_stats"] = data.get("mutation_stats", {})
                self.metrics["attack_totals"] = data.get("attack_totals", {})
                self.metrics["mutation_totals"] = data.get("mutation_totals", {})
                gens = data.get("generations", [])
                if gens:
                    self.metrics["latest_gen"] = max(gens)

                # Use totals if present, otherwise fall back to aggregating stats
                if self.metrics["attack_totals"]:
                    self.attack_totals = dict(self.metrics["attack_totals"])
                else:
                    self.attack_totals.clear()
                    for gen_stats in self.metrics["attack_stats"].values():
                        for name, count in gen_stats.items():
                            self.attack_totals[name] = self.attack_totals.get(name, 0) + count

                if self.metrics["mutation_totals"]:
                    self.mutation_totals = dict(self.metrics["mutation_totals"])
                else:
                    self.mutation_totals.clear()
                    for gen_stats in self.metrics["mutation_stats"].values():
                        for name, count in gen_stats.items():
                            self.mutation_totals[name] = self.mutation_totals.get(name, 0) + count
            except Exception:
                pass

    def draw(self, stdscr):
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        title = "KernelHunter Attack/Muation Stats"
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)
        stdscr.addstr(1, 2, f"Latest Generation: {self.metrics['latest_gen']}")
        row = 3

        attack = self.attack_totals
        mut = self.mutation_totals

        stdscr.addstr(row, 2, "Attack Types", curses.A_UNDERLINE)
        stdscr.addstr(row, width // 2, "Mutation Types", curses.A_UNDERLINE)
        row += 1

        attack_items = sorted(attack.items(), key=lambda x: x[1], reverse=True)
        mut_items = sorted(mut.items(), key=lambda x: x[1], reverse=True)
        max_rows = height - row - 2

        half_width = width // 2 - 2
        self._draw_items_in_columns(
            stdscr,
            row,
            2,
            attack_items,
            half_width,
            max_rows,
        )
        self._draw_items_in_columns(
            stdscr,
            row,
            width // 2,
            mut_items,
            half_width,
            max_rows,
        )

        stdscr.addstr(height - 1, 0, "Press 'q' to quit")
        stdscr.refresh()

    def run(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(200)
        self.load_metrics()
        while True:
            if time.time() - self.last_refresh >= 1:
                self.load_metrics()
                self.last_refresh = time.time()
            self.draw(stdscr)
            key = stdscr.getch()
            if key == ord('q'):
                break


def main():
    monitor = AttackMutationMonitor()
    curses.wrapper(monitor.run)

if __name__ == "__main__":
    main()
