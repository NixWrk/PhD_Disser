param(
    [Parameter(Mandatory = $true)]
    [string]$Notebook,
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$IPythonDir = Join-Path $RepoRoot ".venv\ipython"
$JupyterRuntimeDir = Join-Path $RepoRoot ".venv\jupyter-runtime"
$NotebookPath = (Resolve-Path -LiteralPath $Notebook).Path
$RepoPrefix = $RepoRoot.TrimEnd("\") + "\"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Missing .venv; run tools/bootstrap.ps1 first"
}
if (-not $NotebookPath.StartsWith($RepoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Notebook must be inside the repository"
}
if ([System.IO.Path]::GetExtension($NotebookPath) -ne ".ipynb") {
    throw "Expected an .ipynb file"
}

$null = New-Item -ItemType Directory -Force -Path $IPythonDir, $JupyterRuntimeDir
$env:IPYTHONDIR = $IPythonDir
$env:JUPYTER_RUNTIME_DIR = $JupyterRuntimeDir

& $VenvPython -m nbconvert --to notebook --execute --inplace $NotebookPath `
    "--ExecutePreprocessor.timeout=$TimeoutSeconds" `
    "--ExecutePreprocessor.kernel_name=breathgeom"
if ($LASTEXITCODE -ne 0) {
    throw "Notebook execution failed: $NotebookPath"
}
