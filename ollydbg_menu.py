#!/usr/bin/env python3
"""Simple curses interface inspired by OllyDbg to run KernelHunter tools."""
import curses
import subprocess
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MENU_ITEMS = [
    ("Run KernelHunter Fuzzer", "kernelHunter.py"),
    ("Crash Explorer", "kernel_crash_ui.py"),
    ("Reservoir Manager", "reservoir_ui.py"),
    ("KernelHunter Monitor", "kernel_hunter_monitor.py"),
    ("Attack/Mutation Stats", "attack_mutation_monitor.py"),
    ("Kernel Error Dashboard", "kernel_dash.py"),
    ("Quit", None),
]

def draw_layout(stdscr, selected_idx):
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    # Divide screen into menu (left) and output (right) panels
    menu_width = max(20, width // 4)
    output_width = width - menu_width - 1

    # Draw menu border
    stdscr.vline(0, menu_width, curses.ACS_VLINE, height)
    stdscr.addstr(0, 1, "KernelHunter")
    for idx, (label, _) in enumerate(MENU_ITEMS):
        y = 2 + idx
        if idx == selected_idx:
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(y, 1, label[:menu_width-2])
            stdscr.attroff(curses.A_REVERSE)
        else:
            stdscr.addstr(y, 1, label[:menu_width-2])
    stdscr.refresh()
    return menu_width


def display_output(stdscr, menu_width, lines):
    height, width = stdscr.getmaxyx()
    output_width = width - menu_width - 1
    for i, line in enumerate(lines[-(height-2):]):
        stdscr.addstr(1 + i, menu_width + 1, line[:output_width-1])
    stdscr.refresh()


def run_program(script_path):
    try:
        result = subprocess.run([
            "python3", script_path
        ], cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = result.stdout.splitlines()
    except Exception as e:
        output = [f"Error running {script_path}: {e}"]
    if not output:
        output = ["(no output)"]
    return output


def main(stdscr):
    curses.curs_set(0)
    stdscr.keypad(True)
    selected_idx = 0
    output_lines = []
    while True:
        menu_width = draw_layout(stdscr, selected_idx)
        display_output(stdscr, menu_width, output_lines)
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord('k')):
            selected_idx = (selected_idx - 1) % len(MENU_ITEMS)
        elif key in (curses.KEY_DOWN, ord('j')):
            selected_idx = (selected_idx + 1) % len(MENU_ITEMS)
        elif key in (curses.KEY_ENTER, ord('\n')):
            label, script = MENU_ITEMS[selected_idx]
            if label == "Quit":
                break
            script_path = os.path.join(BASE_DIR, script)
            curses.endwin()
            output_lines = run_program(script_path)
            stdscr = curses.initscr()
            curses.curs_set(0)
            stdscr.keypad(True)
        elif key in (ord('q'), ord('Q')):
            break

curses.wrapper(main)
