#!/usr/bin/env bash
# One command so every student runs the notebook in the SAME local kernel.
#
#   bash env/setup_local_env.sh
#
# Creates .venv, installs the (tiny) notebook requirements, and registers a
# kernel called "Nautilus Lab (.venv)". In VS Code / Jupyter, pick that kernel.
# This also avoids a trap we hit for real: a kernel left registered by an old
# project whose virtualenv was deleted. Selecting it makes the FIRST cell fail
# with a kernel-startup error that looks like a notebook bug but is not.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
"$PY" -m venv .venv
./.venv/bin/python -m pip install --quiet --upgrade pip
./.venv/bin/python -m pip install --quiet -r env/requirements-notebook.txt
./.venv/bin/python -m ipykernel install --user \
    --name nautilus-lab --display-name "Nautilus Lab (.venv)"

echo
echo "kernel registered: 'Nautilus Lab (.venv)'  ($(./.venv/bin/python -V))"
echo "Select it in VS Code (top-right 'Select Kernel') before running the notebook."
echo
echo "Kernels registered on this machine — a 'DEAD' one must not be selected:"
./.venv/bin/python - <<'PY'
import json, os, subprocess
out = subprocess.run(["jupyter", "kernelspec", "list", "--json"],
                     capture_output=True, text=True).stdout
for name, spec in json.loads(out or '{"kernelspecs":{}}')["kernelspecs"].items():
    argv = spec["spec"]["argv"]
    interp = argv[0]
    state = "ok  " if (os.path.exists(interp) or "/" not in interp) else "DEAD"
    print(f"  [{state}] {name:<18} {interp}")
PY
