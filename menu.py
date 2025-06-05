#!/usr/bin/env python3
"""Simple curses menu to launch KernelHunter tools."""

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


def draw_menu(stdscr, selected):
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    title = "KernelHunter Menu"
    stdscr.addstr(1, max(0, width // 2 - len(title) // 2), title, curses.A_BOLD)
    for idx, (label, _) in enumerate(MENU_ITEMS):
        y = 3 + idx
        x = max(0, width // 2 - len(label) // 2)
        if idx == selected:
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(y, x, label)
            stdscr.attroff(curses.A_REVERSE)
        else:
            stdscr.addstr(y, x, label)
    stdscr.refresh()


def run_menu(stdscr):
    curses.curs_set(0)
    stdscr.keypad(True)
    idx = 0
    while True:
        draw_menu(stdscr, idx)
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord('k')):
            idx = (idx - 1) % len(MENU_ITEMS)
        elif key in (curses.KEY_DOWN, ord('j')):
            idx = (idx + 1) % len(MENU_ITEMS)
        elif key in (curses.KEY_ENTER, ord('\n')):
            label, script = MENU_ITEMS[idx]
            if label == "Quit":
                break
            curses.endwin()
            script_path = os.path.join(BASE_DIR, script)
            subprocess.call(["python3", script_path], cwd=BASE_DIR)
            stdscr = curses.initscr()
            curses.curs_set(0)
            stdscr.keypad(True)
        elif key in (ord('q'), ord('Q')):
            break


def main():
    curses.wrapper(run_menu)


if __name__ == "__main__":
    main()
