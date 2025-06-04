#!/usr/bin/env python3
import curses
import json
import os
import time

METRICS_FILE = "kernelhunter_metrics.json"

class AttackMutationMonitor:
    def __init__(self):
        self.metrics = {
            "attack_stats": {},
            "mutation_stats": {},
            "latest_gen": 0,
        }
        # Totals accumulated across all generations
        self.attack_totals = {}
        self.mutation_totals = {}
        self.last_refresh = 0

    def load_metrics(self):
        if os.path.exists(METRICS_FILE):
            try:
                with open(METRICS_FILE, 'r') as f:
                    data = json.load(f)
                self.metrics["attack_stats"] = data.get("attack_stats", {})
                self.metrics["mutation_stats"] = data.get("mutation_stats", {})
                gens = data.get("generations", [])
                if gens:
                    self.metrics["latest_gen"] = max(gens)

                # Recalculate totals each time metrics are loaded
                self.attack_totals.clear()
                for gen_stats in self.metrics["attack_stats"].values():
                    for name, count in gen_stats.items():
                        self.attack_totals[name] = self.attack_totals.get(name, 0) + count

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
        for i in range(max_rows):
            if i < len(attack_items):
                a_name, a_count = attack_items[i]
                stdscr.addstr(row + i, 2, f"{a_name}: {a_count}")
            if i < len(mut_items):
                m_name, m_count = mut_items[i]
                stdscr.addstr(row + i, width // 2, f"{m_name}: {m_count}")

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
