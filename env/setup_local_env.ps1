# Windows (PowerShell) equivalent of setup_local_env.sh — same result:
# a .venv and a kernel named "Nautilus Lab (.venv)" that you select in VS Code.
#
#   powershell -ExecutionPolicy Bypass -File env\setup_local_env.ps1
#
# Note: kubectl needs its kubelogin plugin on PATH to authenticate. On Windows:
#   winget install -e --id Kubernetes.kubectl
#   winget install -e --id int128.kubelogin
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $py -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
& .\.venv\Scripts\python.exe -m pip install --quiet -r env\requirements-notebook.txt
& .\.venv\Scripts\python.exe -m ipykernel install --user `
    --name nautilus-lab --display-name "Nautilus Lab (.venv)"

Write-Host ""
Write-Host "kernel registered: 'Nautilus Lab (.venv)'"
Write-Host "Select it in VS Code (top-right 'Select Kernel') before running the notebook."
Write-Host ""
Write-Host "Kernels registered on this machine — a 'DEAD' one must not be selected:"
& .\.venv\Scripts\python.exe -c @"
import os
from jupyter_client.kernelspec import KernelSpecManager
for name, sp in KernelSpecManager().get_all_specs().items():
    interp = sp['spec']['argv'][0]
    state = 'ok  ' if (os.path.exists(interp) or os.sep not in interp) else 'DEAD'
    print(f'  [{state}] {name:<18} {interp}')
"@
