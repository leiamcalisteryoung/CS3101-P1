#!/bin/sh
# run.sh - USQL interpreter entry point
#
# Usage:
#   sh run.sh program.usql        → interpret and print final query result
#   sh run.sh --o program.usql    → run query optimiser (prints optimisation steps)

# Resolve the directory this script lives in so src/ imports work regardless
# of where the user invokes run.sh from.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Run the interpreter, keeping the current working directory so that relative
# LOAD paths in USQL programs (e.g. "./module.csv") resolve correctly.
# Use the venv python if present (has lark installed), otherwise fall back.
if [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python3"
else
    PYTHON="python3"
fi

PYTHONPATH="$SCRIPT_DIR/src" "$PYTHON" "$SCRIPT_DIR/src/main.py" "$@"
