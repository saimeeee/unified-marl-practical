param(
    [string]$PythonPath = "",
    [ValidateSet("core", "warehouse", "warehouse-role", "traffic", "powergrid", "reward", "role", "hmf", "distributed")]
    [string]$Feature = "core",
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$bundleRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $bundleRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) { throw "Existing engineering .venv required. Install the base bundle first." }
$featureArgs = @("-X", "utf8", "-I", (Join-Path $bundleRoot ".unified-marl-hotfix\0.8.2\feature_install.py"), "--bundle", $bundleRoot, "--feature", $Feature)
if ($DryRun) { $featureArgs += "--dry-run" }
& $venvPython @featureArgs
exit $LASTEXITCODE
