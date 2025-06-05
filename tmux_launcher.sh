#!/bin/bash

# Ensure tmux is installed
if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is not installed. Please install tmux to run this launcher." >&2
    exit 1
fi

# Resolve the directory of this script and switch to it
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SESSION="kernelhunter"

# Start tmux session and launch the monitoring tools
# First pane: KernelHunter monitor
 tmux new-session -d -s "$SESSION" "python3 kernel_hunter_monitor.py"

# Second pane: Reservoir UI (split horizontally)
 tmux split-window -h -t "$SESSION" "python3 reservoir_ui.py"

# Third pane: Kernel crash UI (split vertically from pane 0)
 tmux select-pane -t "$SESSION":0.0
 tmux split-window -v -t "$SESSION" "python3 kernel_crash_ui.py"

# Fourth pane: Kernel dashboard (split vertically from pane 1)
 tmux select-pane -t "$SESSION":0.1
 tmux split-window -v -t "$SESSION" "python3 kernel_dash.py"

# Arrange panes in a tiled layout (2 columns, 2 rows)
 tmux select-layout -t "$SESSION" tiled

# Attach to the session
 tmux attach-session -t "$SESSION"

