$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $repoRoot ".venv-registration"
$pythonPath = Join-Path $environmentPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    py -3.11 -m venv $environmentPath
}

& $pythonPath -m pip install `
    torch==2.5.1 `
    --index-url https://download.pytorch.org/whl/cu118

& $pythonPath -m pip install `
    convexAdam==0.2.0 `
    numpy==1.26.4 `
    scipy==1.13.1 `
    nibabel==5.3.2 `
    scikit-learn==1.5.2 `
    SimpleITK==2.5.2

& $pythonPath -c @"
import torch
import convexAdam
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("convexAdam", convexAdam.__file__)
"@
