#!/usr/bin/env python3
"""Generate a detailed KernelHunter report.

This module summarizes metrics collected during KernelHunter fuzzing
runs, reservoir diversity statistics, and crash information. The
resulting report is saved as a text file for offline analysis.
"""

import json
import os
import pickle
import curses
from datetime import datetime
from statistics import mean

try:
    from kernelhunter_config import get_reservoir_file
except Exception:
    def get_reservoir_file(name: str = "kernelhunter_reservoir.pkl") -> str:
        return name

METRICS_FILE = "kernelhunter_metrics.json"
CRITICAL_DIR = "kernelhunter_critical"
CRASH_DIR = "kernelhunter_crashes"
REPORT_FILE = "kernelhunter_report.txt"


def load_metrics(path: str = METRICS_FILE) -> dict:
    """Load metrics from JSON file if available."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_reservoir_stats(reservoir_path: str) -> dict:
    """Load diversity statistics from the genetic reservoir."""
    if not os.path.exists(reservoir_path):
        return {}

    try:
        from genetic_reservoir import GeneticReservoir

        reservoir = GeneticReservoir()
        if reservoir.load_from_file(reservoir_path):
            return reservoir.get_diversity_stats()
        # Fallback: attempt direct pickle
        with open(reservoir_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, GeneticReservoir):
            return data.get_diversity_stats()
    except Exception:
        try:
            with open(reservoir_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "reservoir" in data:
                size = len(data.get("reservoir", []))
                return {"reservoir_size": size}
        except Exception:
            pass
    return {}


def count_crash_files() -> int:
    """Count JSON crash files across crash directories."""
    total = 0
    for d in (CRITICAL_DIR, CRASH_DIR):
        if os.path.exists(d):
            total += len([f for f in os.listdir(d) if f.endswith(".json")])
    return total


def summarize_metrics(metrics: dict) -> dict:
    """Compute high level summaries from raw metrics."""
    summary = {}
    gens = metrics.get("generations", [])
    if gens:
        summary["total_generations"] = max(gens)
    crash_rates = metrics.get("crash_rates", [])
    if crash_rates:
        summary["average_crash_rate"] = mean(crash_rates)
    system_impacts = metrics.get("system_impacts", [])
    if system_impacts:
        summary["total_system_impacts"] = sum(system_impacts)
    lengths = metrics.get("shellcode_lengths", [])
    if lengths:
        summary["average_shellcode_length"] = mean(lengths)
    summary["attack_totals"] = metrics.get("attack_totals", {})
    summary["mutation_totals"] = metrics.get("mutation_totals", {})
    summary["crash_types"] = metrics.get("crash_types", {})
    return summary


def generate_report(output_path: str = REPORT_FILE) -> None:
    """Generate the report and write it to *output_path*."""
    metrics = load_metrics()
    reservoir_path = get_reservoir_file()
    reservoir_stats = load_reservoir_stats(reservoir_path)
    crash_count = count_crash_files()
    summary = summarize_metrics(metrics)

    lines = []
    lines.append("=" * 80)
    lines.append("KERNELHUNTER FUZZING REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Metrics summary
    lines.append("[Metrics]")
    if summary:
        if "total_generations" in summary:
            lines.append(f"Total generations: {summary['total_generations']}")
        if "average_crash_rate" in summary:
            lines.append(
                f"Average crash rate: {summary['average_crash_rate']:.2%}")
        if "total_system_impacts" in summary:
            lines.append(
                f"Total system impacts: {summary['total_system_impacts']}")
        if "average_shellcode_length" in summary:
            lines.append(
                f"Average shellcode length: {summary['average_shellcode_length']:.1f} bytes")
    else:
        lines.append("No metrics data available.")
    lines.append("")

    # Crash info
    lines.append("[Crashes]")
    lines.append(f"Crash dataset size: {crash_count} JSON files")
    if summary.get("crash_types"):
        latest_key = str(summary.get("total_generations", ""))
        crash_types = summary["crash_types"].get(latest_key) or summary["crash_types"].get(int(latest_key))
        if crash_types:
            lines.append("Latest generation crash types:")
            for name, count in crash_types.items():
                lines.append(f"  - {name}: {count}")
    lines.append("")

    # Attack and mutation totals
    if summary.get("attack_totals") or summary.get("mutation_totals"):
        lines.append("[Evolution Stats]")
        if summary.get("attack_totals"):
            lines.append("Attack totals:")
            for name, count in sorted(summary["attack_totals"].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  - {name}: {count}")
        if summary.get("mutation_totals"):
            lines.append("Mutation totals:")
            for name, count in sorted(summary["mutation_totals"].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  - {name}: {count}")
        lines.append("")

    # Reservoir stats
    lines.append("[Genetic Reservoir]")
    if reservoir_stats:
        for key, value in reservoir_stats.items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
    else:
        lines.append("Reservoir statistics not available.")
    lines.append("")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written to {output_path}")


def view_report_curses(stdscr, report_path: str = REPORT_FILE):
    """Simple curses UI to view and regenerate the report."""
    curses.curs_set(0)
    offset = 0
    while True:
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = ["Report not found. Press 'r' to generate it."]

        height, width = stdscr.getmaxyx()
        max_lines = height - 2
        offset = max(0, min(offset, max(0, len(lines) - max_lines)))

        stdscr.clear()
        for i in range(max_lines):
            if offset + i >= len(lines):
                break
            stdscr.addstr(i, 0, lines[offset + i][: width - 1])

        stdscr.addstr(height - 1, 0, "↑↓ scroll  r: regenerate  q: quit")
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP and offset > 0:
            offset -= 1
        elif key == curses.KEY_DOWN and offset < len(lines) - max_lines:
            offset += 1
        elif key == ord('q'):
            break
        elif key == ord('r'):
            generate_report(report_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a KernelHunter report")
    parser.add_argument("--output", default=REPORT_FILE, help="Output report file")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch a curses interface to view and regenerate the report",
    )
    args = parser.parse_args()
    generate_report(args.output)
    if args.interactive:
        curses.wrapper(view_report_curses, args.output)
