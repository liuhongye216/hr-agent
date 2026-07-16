param(
    [ValidateRange(1, 500)]
    [int]$Count = 100,
    [string]$OutputDir = (Join-Path $PSScriptRoot "data")
)

$ErrorActionPreference = "Stop"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv (Join-Path $PSScriptRoot ".venv")
}

& $venvPython -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot "requirements.txt")
& (Join-Path $PSScriptRoot "start_browser.ps1")

Read-Host "Complete any visible BOSS security check in the browser, then press Enter"

& $venvPython -u (Join-Path $PSScriptRoot "collector.py") `
    --count $Count `
    --cdp-url "http://127.0.0.1:9222" `
    --output-dir $OutputDir

if ($LASTEXITCODE -eq 2) {
    Write-Host "Collection is incomplete. Resolve any visible security check and run this script again to resume."
}

exit $LASTEXITCODE
